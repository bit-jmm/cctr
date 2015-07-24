#include "urrp.h"

using namespace std;

inline double square(double x)
{
  return x * x;
}

void URRP::init_model()
{
  for(vector<rating*>::iterator it=trainratings.begin(); it != trainratings.end(); it++)
  {
    int user = (*it)->user;
    int item = (*it)->item;
    int rating = (*it)->value;
    vector<int>* words = &((*it)->words);

    //init attitude and topic
    int attitude = rand() % K;
    (*x)(user, item) = attitude;
    (*muk)(user, attitude)++;
    (*ckvs)(attitude, item)(rating)++;

    int topic = rand() % K;
    for(vector<int>::iterator it2=words->begin(); it2 != words->end(); it2++)
    {
      (*z)(user, item)(*it2) = topic;
      (*nuk)(user, topic)++;
      (*nkw)(topic, *it2)++;
      topic = rand() % K;
    }
  }
}
void URRP::sample_topic_attitude_assignments()
{

}

bool URRP::is_converged()
{
  return false;
}

double URRP::get_nk(int k)
{
  double val = 0.0;
  for(int w=0; w<nWords; w++)
  {
    val += (*nkw)(k, w);
  }
  return val;
}

double URRP::get_nu(int u)
{
  double val = 0.0;
  for(int k=0; k<K; k++)
  {
    val += (*nuk)(u, k);
  }
  return val;
}

double URRP::get_mu(int u)
{
  double val = 0.0;
  for(int k=0; k<K; k++)
  {
    val += (*muk)(u, k);
  }
  return val;
}

double URRP::get_ckv(int k, int v)
{
  double val = 0.0;
  for(int s=0; s<S; s++)
  {
    val += (*ckvs)(k, v)(s);
  }
  return val;
}

void URRP::readout_params()
{
  double nu = 0.0;
  double mu = 0.0;
  double nk = 0.0;
  double ckv = 0.0;

  for(int u=0; u<nUsers; u++)
  {
    nu = get_nu(u);
    mu = get_mu(u);
    for(int k=0; k<K; k++)
    {
       (*theta)(u, k) = ((*nuk)(u, k) + (*muk)(u, k) + alpha[k]) / (nu + mu + sum_alpha);
    }
  }

  for(int k=0; k<K; k++)
  {
    nk = get_nk(k);
    for(int w=0; w<nWords; w++)
    {
       (*phi)(k, w) = ((*nkw)(k, w) + beta[w]) / (nk + sum_beta);
    }
  }

  for(int k=0; k<K; k++)
    for(int v=0; v<nItems; v++)
    {
      ckv = get_ckv(k,v);
      for(int s=0; s<S; s++)
      {
        (*xi)(k, v)(s) = ((*ckvs)(k, v)(s) + lambda[s]) / (ckv + sum_lambda);
      }
    }
}
void URRP::update_hyperparameters()
{

}

/// Train a model
void URRP::train()
{
  init_model();
  readout_params();
  validate_test();
  return;
  for (int iter = 0; iter < max_iter; iter++) {
    // sample topic and attitude for all words and ratings
    sample_topic_attitude_assignments();
    // update hyper-parameters
    update_hyperparameters();

    // get statistics after burn-in
    if ((iter > burn_in) && (iter % sample_lag == 0)) {
      readout_params();
      if (is_converged())
      {
        break;
      }
    }
  }
}

void URRP::validate_test()
{
  double validate_err = 0.0;
  double test_err = 0.0;
  double test_ste = 0.0;
  for(vector<rating*>::iterator it=validratings.begin(); it!=validratings.end(); it++)
  {
    validate_err += square(predict(*it) - (*it)->value);
  }

  for(vector<rating*>::iterator it=testratings.begin(); it!=testratings.end(); it++)
  {
    double err = square(predict(*it) - (*it)->value);
    test_err += err;
    test_ste += err*err;
  }
  validate_err /= validratings.size();
  test_err /= testratings.size();
  test_ste /= testratings.size();
  test_ste = sqrt((test_ste-test_err*test_err)/testratings.size());

  printf("\nValidation MSE: %.4lf, Test MSE: %.4lf (%.2lf)\n", validate_err, test_err, test_ste);
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
      ps += (*theta)(user, k) * (*xi)(k, item)(s);
    }
    pred += ps * (s+1);
  }
  return pred;
}
