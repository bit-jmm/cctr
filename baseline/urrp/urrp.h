#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <fstream>
using namespace Eigen;
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

  //attitude assignments for each rating r_u,v
  MatrixXd * x;
  //topic assignments for each word in d_u,v,w
  Matrix<VectorXd, Dynamic, Dynamic> * z;

  // times that topic k has been assigned to word w
  MatrixXd * nkw;

  // times that topic k has been assigned to words in reviews written by user u
  MatrixXd * nuk;
  // times that attitude k assigned to ratings by user u
  MatrixXd * muk;
  // times that rating s assigned to item v when attitude is k
  Matrix<VectorXd, Dynamic, Dynamic> * ckvs;

  MatrixXf * theta;
  MatrixXf * phi;
  Matrix<VectorXf, Dynamic, Dynamic> * xi;

  double sum_alpha;
  double sum_beta;
  double sum_lambda;

  double mse;

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

    this->x = new MatrixXd(nUsers, nItems);
    this->z = new Matrix<VectorXd, Dynamic, Dynamic>(nUsers, nItems);
    for(int u=0; u<nUsers; u++)
      for(int v=0; v<nItems; v++)
        (*z)(u, v) = VectorXd(nWords);

    this->nkw = new MatrixXd(K, nWords);
    this->nuk = new MatrixXd(nUsers,K);
    this->muk = new MatrixXd(nUsers, K);
    this->ckvs = new Matrix<VectorXd, Dynamic, Dynamic>(K, nItems);
    for(int k=0; k<K; k++)
      for(int v=0; v<nItems; v++)
        (*ckvs)(k, v) = VectorXd(S);

    this->theta = new MatrixXf(nUsers, K);
    this->phi = new MatrixXf(K, nWords);
    this->xi = new Matrix<VectorXf, Dynamic, Dynamic>(K, nItems);
    for(int k=0; k<K; k++)
      for(int v=0; v<nItems; v++)
        (*xi)(k, v) = VectorXf(S);

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
  bool is_converged();
  double get_nk(int k);
  double get_nu(int u);
  double get_mu(int u);
  double get_ckv(int k, int v);
  void readout_params();
  void update_hyperparameters();
  void validate_test();
  double predict(rating* vi);
  void train();

  ~URRP()
  {
    delete[] alpha;
    delete[] beta;
    delete[] lambda;
    delete[] x;
    delete[] z;
    delete[] nkw;
    delete[] nuk;
    delete[] muk;
    delete[] ckvs;
    delete[] theta;
    delete[] phi;
    delete[] xi;
  }

};
