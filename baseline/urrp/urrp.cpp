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

    int topic = rand() % K;
    for(vector< pair<int,int> >::iterator it2=words->begin(); it2 != words->end(); it2++)
    {
      (*it2).second = topic;
      (*nuk)[user][topic]++;
      (*nkw)[topic][(*it2).first]++;
      topic = rand() % K;
    }
  }
}
void URRP::sample_topic_attitude_assignments()
{
  double* p = new double[K];
  for(vector<rating*>::iterator it=trainratings.begin(); it != trainratings.end(); it++)
  {
    int user = (*it)->user;
    int item = (*it)->item;
    int rating = (*it)->value;
    vector< pair<int,int> >* words = &((*it)->words);

    //delete related attitude and topic for this rating;
    int current_attitude = (*it)->attitude;
    (*muk)[user][current_attitude]--;
    (*ckvs)[current_attitude][item][rating]--;

    int nu = get_nu(user);
    int mu = get_mu(user);
    int k=0;
    #pragma omp parallel for
    for(k=0; k<K; k++)
    {
      p[k] = ((*nuk)[user][k] + (*muk)[user][k] + alpha[k]) / (nu + mu + sum_alpha) * (((*ckvs)[k][item][rating] + lambda[rating]) / (get_ckv(k, item) + sum_lambda));
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
    if(k==K) k = K-1;
    (*it)->attitude = k;
    (*muk)[user][k]++;
    (*ckvs)[k][item][rating]++;

    nu = get_nu(user);
    mu = get_mu(user);

    for(vector< pair<int,int> >::iterator it2=words->begin(); it2 != words->end(); it2++)
    {
      int w = (*it2).first;
      int current_topic = (*it2).second;
      (*nuk)[user][current_topic]--;
      (*nkw)[current_topic][w]--;
      int k=0;
      #pragma omp parallel for
      for(k=0; k<K; k++)
      {
        p[k] = (((*nuk)[user][k] + (*muk)[user][k] + alpha[k]) / (nu + mu + sum_alpha) * (((*nkw)[k][w] + beta[w]) / (get_nk(k) + sum_beta)));
      }
      for(k=1; k<K; k++)
        p[k] += p[k-1];

      randdouble = (rand() * 1.0 / RAND_MAX) * p[K-1];
      for(k=0; k<K; k++)
      {
        if(randdouble <= p[k])
        {
          break;
        }
      }
      if(k==K) k = K-1;
      (*it2).second = k;
      (*nuk)[user][k]++;
      (*nkw)[k][w]++;
    }
  }
  delete[] p;
}

int URRP::get_nk(int k)
{
  int val = 0, w;
  #pragma omp parallel for reduction (+:val)
  for(w=0; w<nWords; w++)
  {
    val += (*nkw)[k][w];
  }
  return val;
}

int URRP::get_nu(int u)
{
  int val = 0, k;
  #pragma omp parallel for reduction (+:val)
  for(k=0; k<K; k++)
  {
    val += (*nuk)[u][k];
  }
  return val;
}

int URRP::get_mu(int u)
{
  int val = 0, k;
  #pragma omp parallel for reduction (+:val)
  for(k=0; k<K; k++)
  {
    val += (*muk)[u][k];
  }
  return val;
}

int URRP::get_ckv(int k, int v)
{
  int val = 0, s;
  #pragma omp parallel for reduction (+:val)
  for(s=0; s<S; s++)
  {
    val += (*ckvs)[k][v][s];
  }
  return val;
}

void URRP::readout_params()
{
  int u,v,k,w,s;
#pragma omp parallel for collapse(2)
  for(u=0; u<nUsers; u++)
    for(k=0; k<K; k++)
      (*theta)[u][k] = ((*nuk)[u][k] + (*muk)[u][k] + alpha[k]) / (get_nu(u) + get_mu(u) + sum_alpha);

#pragma omp parallel for collapse(2)
  for(k=0; k<K; k++)
    for(w=0; w<nWords; w++)
      (*phi)[k][w] = ((*nkw)[k][w] + beta[w]) / (get_nk(k) + sum_beta);

#pragma omp parallel for collapse(3)
  for(k=0; k<K; k++)
    for(v=0; v<nItems; v++)
      for(s=0; s<S; s++)
        (*xi)[k][v][s] = ((*ckvs)[k][v][s] + lambda[s]) / (get_ckv(k,v) + sum_lambda);
}
void URRP::update_hyperparameters()
{
  int u,v,k,w,s;
  double ak, betaw,lambdas, numerator, denominator;
  //update alpha
  for (k = 0; k < K; k++) {
    ak = alpha[k];
    numerator = 0, denominator = 0;
    #pragma omp parallel for reduction (+:numerator,denominator)
    for (u = 0; u < nUsers; u++) {
      numerator += digamma((*nuk)[u][k] + (*muk)[u][k] + ak) - digamma(ak);
      denominator += digamma(get_nu(u) + get_mu(u) + sum_alpha) - digamma(sum_alpha);
    }
    if (numerator != 0)
      alpha[k] = ak * (numerator / denominator);
  }
  sum_alpha = 0;
  for (k = 0; k < K; k++)
  {
    sum_alpha += alpha[k];
  }
  //update beta
  for (w = 0; w < nWords; w++) {
    betaw = beta[w];
    numerator = 0, denominator = 0;
    #pragma omp parallel for reduction (+:numerator,denominator)
    for (k = 0; k < K; k++) {
      numerator += digamma((*nkw)[k][w] + betaw) - digamma(betaw);
      denominator += digamma(get_nk(k) + sum_beta) - digamma(sum_beta);
    }
    if (numerator != 0)
      beta[w] = betaw * (numerator / denominator);
  }
  double tmp_sum_beta = 0;
  #pragma omp parallel for reduction (+:tmp_sum_beta)
  for (w = 0; w < nWords; w++)
  {
    tmp_sum_beta += beta[w];
  }
  sum_beta = tmp_sum_beta;
  //update lambda
  for (s = 0; s < S; s++) {
    lambdas = lambda[s];
    numerator = 0, denominator = 0;
    #pragma omp parallel for collapse(2) reduction (+:numerator,denominator)
    for(k=0; k<K; k++)
      for(v=0; v<nItems; v++)
      {
        numerator += digamma((*ckvs)[k][v][s] + lambdas) - digamma(lambdas);
        denominator += digamma(get_ckv(k, v) + sum_lambda) - digamma(sum_lambda);
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

bool URRP::is_converged(int iter)
{
  readout_params();
  double validate_err = 0.0;
  double test_err = 0.0;
  double test_ste = 0.0;
  size_t i;

  #pragma omp parallel for reduction (+:validate_err)
  for(i=0; i<validratings.size(); i++)
  {
    validate_err += square(predict(validratings[i]) - validratings[i]->value - 1);
  }

  #pragma omp parallel for reduction (+:test_err,test_ste)
  for(i=0; i<testratings.size(); i++)
  {
    double err = square(predict(testratings[i]) - testratings[i]->value - 1);
    test_err += err;
    test_ste += err*err;
  }
  validate_err /= validratings.size();
  test_err /= testratings.size();
  test_ste /= testratings.size();
  test_ste = sqrt((test_ste-test_err*test_err)/testratings.size());

  double delta = validate_err - prev_mse;
  printf("\nIter: %d, Validation MSE: %.4lf, Test MSE: %.4lf (%.2lf), validate_mse_delta: %.4lf\n", iter, validate_err, test_err, test_ste, delta);

  if (delta > 0)
  {
    return true;
  }
  prev_mse = validate_err;
  return false;
}

/// Train a model
void URRP::train()
{
  init_model();
  is_converged(-1);
  for (int iter = 0; iter < max_iter; iter++) {
    // sample topic and attitude for all words and ratings
    sample_topic_attitude_assignments();
    // update hyper-parameters
    update_hyperparameters();

    // get statistics after burn-in
    if ((iter > burn_in) && (iter % sample_lag == 0))
    {
      if (is_converged(iter) && false)
      {
        break;
      }
    }
    //if (iter % 5 == 0)
    //{
      //printf("iter: %d\n", iter);
    //}
  }
}

// Predict a particular rating given the current parameter values
double URRP::predict(rating* vi)
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
