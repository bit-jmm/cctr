#include "urrp.h"
#include <boost/math/special_functions/digamma.hpp>
using namespace std;

inline double square(double x)
{
  return x * x;
}

inline double digamma(double x)
{
  if(x==0.0)
  {
    return 0.0;
  } else {
    return boost::math::digamma(x);
  }
}

void URRP::init_model()
{
  for(vector<rating*>::iterator it=trainratings.begin(); it != trainratings.end(); it++)
  {
    int user = (*it)->user;
    int item = (*it)->item;
    int rating = (*it)->value;
    vector< pair<int,int> >* words = &((*it)->words);

    //init attitude and topic
    int attitude = rand() % K;
    (*it)->attitude = attitude;
    (*muk)[user][attitude]++;
    (*ckvs)[attitude][item][rating]++;
    (*mu)[user]++;
    (*ckv)[attitude][item]++;

    int topic = rand() % K;
    for(vector< pair<int,int> >::iterator it2=words->begin(); it2 != words->end(); it2++)
    {
      (*it2).second = topic;
      (*nuk)[user][topic]++;
      (*nkw)[topic][(*it2).first]++;
      (*nu)[user]++;
      (*nk)[topic]++;
      topic = rand() % K;
    }
  }
}

void URRP::sample_attitudes(bool with_topic)
{
  double* p = new double[K];
  for(vector<rating*>::iterator it=trainratings.begin(); it != trainratings.end(); it++)
  {
    int user = (*it)->user;
    int item = (*it)->item;
    int rating = (*it)->value;

    //delete related attitude and topic for this rating;
    int current_attitude = (*it)->attitude;
    (*muk)[user][current_attitude]--;
    (*ckvs)[current_attitude][item][rating]--;
    (*mu)[user]--;
    (*ckv)[current_attitude][item]--;

    int k=0;
    //#pragma omp parallel for
    for(k=0; k<K; k++)
    {
      if(with_topic)
        p[k] = ((*nuk)[user][k] + (*muk)[user][k] + alpha[k]) / ((*nu)[user] + (*mu)[user] + sum_alpha) * (((*ckvs)[k][item][rating] + lambda[rating]) / ((*ckv)[k][item] + sum_lambda));
      else
        p[k] = ((*muk)[user][k] + alpha[k]) / ((*mu)[user] + sum_alpha) * (((*ckvs)[k][item][rating] + lambda[rating]) / ((*ckv)[k][item] + sum_lambda));
    }

    for(k=1; k<K; k++)
      p[k] += p[k-1];

    double randdouble = (rand() * 1.0 / RAND_MAX) * p[K-1];
    for(k=0; k<K; k++)
    {
      if(randdouble <= p[k])
      {
        break;
      }
    }
    (*it)->attitude = k;
    (*muk)[user][k]++;
    (*ckvs)[k][item][rating]++;
    (*mu)[user]++;
    (*ckv)[k][item]++;
  }
  delete[] p;
}

void URRP::sample_topics(bool with_attitude)
{
  double* p = new double[K];
  for(vector<rating*>::iterator it=trainratings.begin(); it != trainratings.end(); it++)
  {
    int user = (*it)->user;
    vector< pair<int,int> >* words = &((*it)->words);

    for(vector< pair<int,int> >::iterator it2=words->begin(); it2 != words->end(); it2++)
    {
      int w = (*it2).first;
      int current_topic = (*it2).second;
      (*nuk)[user][current_topic]--;
      (*nkw)[current_topic][w]--;
      (*nu)[user]--;
      (*nk)[current_topic]--;
      int k=0;
      for(k=0; k<K; k++)
      {
        if(with_attitude)
          p[k] = (((*nuk)[user][k] + (*muk)[user][k] + alpha[k]) / ((*nu)[user] + (*mu)[user] + sum_alpha) * (((*nkw)[k][w] + beta[w]) / ((*nk)[k] + sum_beta)));
        else
          p[k] = (((*nuk)[user][k] + alpha[k]) / ((*nu)[user] + sum_alpha) * (((*nkw)[k][w] + beta[w]) / ((*nk)[k] + sum_beta)));
      }
      for(k=1; k<K; k++)
        p[k] += p[k-1];

      double randdouble = (rand() * 1.0 / RAND_MAX) * p[K-1];
      for(k=0; k<K; k++)
      {
        if(randdouble <= p[k])
        {
          break;
        }
      }
      (*it2).second = k;
      (*nuk)[user][k]++;
      (*nkw)[k][w]++;
      (*nu)[user]++;
      (*nk)[k]++;
    }
  }
  delete[] p;
}

void URRP::readout_topic_theta(bool with_attitude)
{
  int u,k;
#pragma omp parallel for collapse(2)
  for(u=0; u<nUsers; u++)
    for(k=0; k<K; k++)
      if(with_attitude)
        (*theta)[u][k] = ((*nuk)[u][k] + (*muk)[u][k] + alpha[k]) / ((*nu)[u] + (*mu)[u] + sum_alpha);
      else
        (*theta)[u][k] = ((*nuk)[u][k] + alpha[k]) / ((*nu)[u] + sum_alpha);
}

void URRP::readout_attitude_theta(bool with_topic)
{
  int u,k;
#pragma omp parallel for collapse(2)
  for(u=0; u<nUsers; u++)
    for(k=0; k<K; k++)
      if(with_topic)
        (*theta)[u][k] = ((*nuk)[u][k] + (*muk)[u][k] + alpha[k]) / ((*nu)[u] + (*mu)[u] + sum_alpha);
      else
        (*theta)[u][k] = ((*muk)[u][k] + alpha[k]) / ((*mu)[u] + sum_alpha);
}

void URRP::readout_phi()
{
  int k,w;
#pragma omp parallel for collapse(2)
  for(k=0; k<K; k++)
    for(w=0; w<nWords; w++)
      (*phi)[k][w] = ((*nkw)[k][w] + beta[w]) / ((*nk)[k] + sum_beta);
}

void URRP::readout_xi()
{
  int v,k,s;
#pragma omp parallel for collapse(3)
  for(k=0; k<K; k++)
    for(v=0; v<nItems; v++)
      for(s=0; s<S; s++)
        (*xi)[k][v][s] = ((*ckvs)[k][v][s] + lambda[s]) / ((*ckv)[k][v] + sum_lambda);
}

void URRP::update_alpha_by_topic()
{
  int k,u;
  double ak, numerator, denominator;
  //update alpha
  for (k = 0; k < K; k++) {
    ak = alpha[k];
    numerator = 0, denominator = 0;
    #pragma omp parallel for reduction (+:numerator,denominator)
    for (u = 0; u < nUsers; u++) {
      numerator += digamma((*nuk)[u][k] + ak) - digamma(ak);
      denominator += digamma((*nu)[u] + sum_alpha) - digamma(sum_alpha);
      //numerator += digamma((*nuk)[u][k] + (*muk)[u][k] + ak) - digamma(ak);
      //denominator += digamma(get_nu(u) + get_mu(u) + sum_alpha) - digamma(sum_alpha);
    }
    if (numerator != 0)
      alpha[k] = ak * (numerator / denominator);
  }
  sum_alpha = 0;
  for (k = 0; k < K; k++)
  {
    sum_alpha += alpha[k];
  }
}

void URRP::update_alpha(bool with_topic)
{
  int k,u;
  double ak, numerator, denominator;
  //update alpha
  for (k = 0; k < K; k++) {
    ak = alpha[k];
    numerator = 0, denominator = 0;
    #pragma omp parallel for reduction (+:numerator,denominator)
    for (u = 0; u < nUsers; u++) {
      if (with_topic)
      {
        numerator += digamma((*nuk)[u][k] + (*muk)[u][k] + ak) - digamma(ak);
        denominator += digamma((*nu)[u] + (*mu)[u] + sum_alpha) - digamma(sum_alpha);
      }
      else
      {
        numerator += digamma((*muk)[u][k] + ak) - digamma(ak);
        denominator += digamma((*mu)[u] + sum_alpha) - digamma(sum_alpha);
      }
    }
    if (numerator != 0)
      alpha[k] = ak * (numerator / denominator);
  }
  sum_alpha = 0;
  for (k = 0; k < K; k++)
  {
    sum_alpha += alpha[k];
  }
}

void URRP::update_beta()
{
  int k,w;
  double betaw, numerator, denominator;
  //update beta
  for (w = 0; w < nWords; w++) {
    betaw = beta[w];
    numerator = 0, denominator = 0;
    //#pragma omp parallel for reduction (+:numerator,denominator)
    for (k = 0; k < K; k++) {
      numerator += digamma((*nkw)[k][w] + betaw) - digamma(betaw);
      denominator += digamma((*nk)[k] + sum_beta) - digamma(sum_beta);
    }
    if (numerator != 0)
      beta[w] = betaw * (numerator / denominator);
  }
  double tmp_sum_beta = 0;
  //#pragma omp parallel for reduction (+:tmp_sum_beta)
  for (w = 0; w < nWords; w++)
  {
    tmp_sum_beta += beta[w];
  }
  sum_beta = tmp_sum_beta;

}

void URRP::update_lambda()
{
  int v,k,s;
  double lambdas, numerator, denominator;
  //update lambda
  for (s = 0; s < S; s++) {
    lambdas = lambda[s];
    numerator = 0, denominator = 0;
    #pragma omp parallel for collapse(2) reduction (+:numerator,denominator)
    for(k=0; k<K; k++)
      for(v=0; v<nItems; v++)
      {
        numerator += digamma((*ckvs)[k][v][s] + lambdas) - digamma(lambdas);
        denominator += digamma((*ckv)[k][v] + sum_lambda) - digamma(sum_lambda);
      }
    if (numerator != 0)
      lambda[s] = lambdas * (numerator / denominator);
  }
  sum_lambda = 0;
  for (s = 0; s < S; s++)
  {
    sum_lambda += lambda[s];
  }
}

void URRP::evaluate(int iter)
{
  double train_err = 0.0;
  double validate_err = 0.0;
  double test_err = 0.0;
  double test_ste = 0.0;
  size_t i;

  #pragma omp parallel for reduction (+:validate_err)
  for(i=0; i<trainratings.size(); i++)
  {
    train_err += square(predict_with_expect(trainratings[i]) - trainratings[i]->value - 1);
  }

  #pragma omp parallel for reduction (+:validate_err)
  for(i=0; i<validratings.size(); i++)
  {
    validate_err += square(predict_with_expect(validratings[i]) - validratings[i]->value - 1);
  }

  #pragma omp parallel for reduction (+:test_err,test_ste)
  for(i=0; i<testratings.size(); i++)
  {
    double err = square(predict_with_expect(testratings[i]) - testratings[i]->value - 1);
    test_err += err;
    test_ste += err*err;
  }
  train_err /= trainratings.size();
  validate_err /= validratings.size();
  test_err /= testratings.size();
  test_ste /= testratings.size();
  test_ste = sqrt((test_ste-test_err*test_err)/testratings.size());

  if(test_err < current_best)
  {
    current_best = test_err;
    current_best_ste = test_ste;
  }

  double delta = validate_err - prev_mse;
  printf("\nRecommend: Iter %d, Train MSE: %.4lf, Validation MSE: %.4lf, Test MSE: %.4lf (%.2lf), validate_mse_delta: %.4lf\n", iter, train_err, validate_err, test_err, test_ste, delta);
  printf("\nCurrent best MSE:\t%.4lf (%.2lf)\n", current_best, current_best_ste);
  prev_mse = validate_err;
  fflush(stdout);
}

/// Train a model
void URRP::train()
{
  init_model();
  readout_attitude_theta(false);
  readout_xi();
  evaluate(0);
  // learn topic distribution by lda
  for(int i=1; i<=burn_in; i++)
  {
    sample_topics(false);
    if (i % 10 == 0)
    {
      printf("LDA stage: iter %d\n", i);
      fflush(stdout);
    }
  }
  readout_topic_theta(false);
  readout_phi();
  topic_words();
  for (int iter = 0; iter <= max_iter; iter++) {
    // sample topic and attitude for all words and ratings
    sample_attitudes(false);
    sample_topics(false);
    // update hyper-parameters
    update_alpha(true);
    update_beta();
    update_lambda();
    // get statistics after burn-in
    if (iter % sample_lag == 0)
    {
      readout_attitude_theta(false);
      readout_phi();
      readout_xi();
      evaluate(iter);
    }
  }
  topic_words();
}

bool word_prob_com(pair<int, double> p1, pair<int, double> p2)
{
  return p1.second > p2.second;
}

void URRP::topic_words()
{
  map<int, string> * id2word = &(corp->id2word);
  for(int k=0; k<K; k++)
  {
    vector< pair<int, double> > topic_words;
    for(int w=0; w<nWords; w++)
    {
      topic_words.push_back(make_pair(w, (*phi)[k][w]));
    }
    sort(topic_words.begin(), topic_words.end(), word_prob_com);
    printf("\nTopic %d:", k+1);
    for(int i=0; i<10; i++)
    {
      printf(" %s(%.4lf)", (*id2word)[topic_words[i].first].c_str(), (*phi)[k][topic_words[i].first]);
    }
  }
  printf("\n");
  fflush(stdout);
}

// Predict a particular rating given the current parameter values
double URRP::predict_with_expect(rating* vi)
{
  int user = vi->user;
  int item = vi->item;
  double pred = 0.0;
  double ps = 0.0;
  for(int s=0; s<S; s++)
  {
    ps = 0.0;
    for(int k=0; k<K; k++)
    {
      ps += (*theta)[user][k] * (*xi)[k][item][s];
    }
    pred += ps * (s+1);
  }
  return pred;
}

// Predict a particular rating given the current parameter values
double URRP::predict_with_most_prob(rating* vi)
{
  int user = vi->user;
  int item = vi->item;
  double pred = 0.0;
  double most_prob = 0.0;
  double ps = 0.0;
  for(int s=0; s<S; s++)
  {
    ps = 0.0;
    for(int k=0; k<K; k++)
    {
      ps += (*theta)[user][k] * (*xi)[k][item][s];
    }
    if(ps > most_prob)
    {
      pred = s+1;
      most_prob = ps;
    }
  }
  return pred;
}
