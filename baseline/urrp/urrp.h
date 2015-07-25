#pragma once

#include "utils.h"
#include <fstream>
#include "boost/multi_array.hpp"
using namespace boost;
typedef multi_array<int, 3> ThreeIntArray;
typedef multi_array<int, 2> TwoIntArray;
typedef multi_array<double, 3> ThreeDoubleArray;
typedef multi_array<double, 2> TwoDoubleArray;

class URRP
{
private:
  Corpus* corp;

  int nUsers; // Number of users
  int nItems; // Number of items
  int nWords; // Number of words

  // ratings from the training, validation, and test sets
  std::vector<rating*> trainratings;
  std::vector<rating*> validratings;
  std::vector<rating*> testratings;

  // num of topics or attitudes
  int K;
  // num of rating values
  int S;
  // hyperparameters
  double* alpha;
  double* beta;
  double* lambda;

  int max_iter;
  int burn_in;
  int sample_lag;

  // times that topic k has been assigned to word w
  TwoIntArray* nkw;

  // times that topic k has been assigned to words in reviews written by user u
  TwoIntArray* nuk;
  // times that attitude k assigned to ratings by user u
  TwoIntArray* muk;
  // times that rating s assigned to item v when attitude is k
  ThreeIntArray* ckvs;

  TwoDoubleArray* theta;
  TwoDoubleArray* phi;
  ThreeDoubleArray* xi;

  double sum_alpha;
  double sum_beta;
  double sum_lambda;

  double prev_mse = 5.0;

public:

  URRP(Corpus* corp, int K, double alpha, double beta, double lambda, int max_iter, int burn_in, int sample_lag) :
    corp(corp), K(K), max_iter(max_iter), burn_in(burn_in), sample_lag(sample_lag)
  {
    this->S = 5;
    srand(0);
    this->nUsers = corp->nUsers;
    this->nItems = corp->nItems;
    this->nWords = corp->nWords;

    this->alpha = new double[K];
    this->beta = new double[nWords];
    this->lambda = new double[S];

    this->nkw = new TwoIntArray(extents[K][nWords]);
    this->nuk = new TwoIntArray(extents[nUsers][K]);
    this->muk = new TwoIntArray(extents[nUsers][K]);
    this->ckvs = new ThreeIntArray(extents[K][nItems][S]);

    this->theta = new TwoDoubleArray(extents[nUsers][K]);
    this->phi = new TwoDoubleArray(extents[K][nWords]);
    this->xi = new ThreeDoubleArray(extents[K][nItems][S]);

    printf("\nnum_factors=%d, rating_values=%d, num_words=%d, alpha=%.2f, beta=%.2f, lambda=%.2f, max_iter=%d, burn_in=%d, sample_lag=%d\n",
        K, S, nWords, alpha, beta, lambda, max_iter, burn_in, sample_lag);

    array_init(this->alpha, K, alpha);
    array_init(this->beta, nWords, beta);
    array_init(this->lambda, S, lambda);

    sum_alpha = alpha * K;
    sum_beta = beta * nWords;
    sum_lambda = lambda * S;

    double testFraction = 0.1;
    if (corp->V->size() > 2400000)
    {
      double trainFraction = 2000000.0 / corp->V->size();
      testFraction = (1.0 - trainFraction)/2;
    }
    //std::ofstream trainFile;
    //trainFile.open("../../data/rating_datasets/" + corp->input_filename + "_train.txt", std::ofstream::out);
    //std::ofstream testFile;
    //testFile.open("../../data/rating_datasets/" + corp->input_filename + "_test.txt", std::ofstream::out);
    for (std::vector<rating*>::iterator it = corp->V->begin(); it != corp->V->end(); it ++)
    {
      double r = rand() * 1.0 / RAND_MAX;
      if (r < testFraction)
      {
        testratings.push_back(*it);
        //testFile << (*it)->user << " " << (*it)->item << " " << (*it)->value << " " << (*it)->ratingTime << std::endl;
      }
      else if (r < 2*testFraction)
        validratings.push_back(*it);
      else
      {
        //trainFile << (*it)->user << " " << (*it)->item << " " << (*it)->value << " " << (*it)->ratingTime << std::endl;
        trainratings.push_back(*it);
      }
    }
    printf("\ntrain ratings: %zu, validate ratings: %zu, test ratings: %zu\n", trainratings.size(), validratings.size(), testratings.size());
    //trainFile.close();
    //testFile.close();
  }

  void init_model();
  void sample_topic_attitude_assignments();
  bool is_converged(int iter);
  int get_nk(int k);
  int get_nu(int u);
  int get_mu(int u);
  int get_ckv(int k, int v);
  void readout_params();
  void update_hyperparameters();
  double predict(rating* vi);
  void train();

  ~URRP()
  {
    delete[] alpha;
    delete[] beta;
    delete[] lambda;
    delete nkw;
    delete nuk;
    delete muk;
    delete ckvs;
    delete theta;
    delete phi;
    delete xi;
  }

};
