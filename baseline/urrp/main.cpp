#include <vector>
#include "sys/time.h"
#include <getopt.h>
#include <string>
#include "urrp.h"

using namespace std;

void parse_parameters(const int argc, char* const* argv, vector<string> &inputfiles,
    double &alpha, double &beta, double &lambda, int &K);

int main(int argc, char** argv)
{
  double alpha = 2;
  double beta = 0.5;
  double lambda = 0.5;
  int K = 5;
  int max_iter = 1000;
  int burn_in = 500;
  int sample_lag = 50;

  vector<string> inputfiles;

  parse_parameters(argc, (char * const *)argv, inputfiles, alpha, beta, lambda, K);

  if(inputfiles.empty())
  {
    printf("Please input training files!\n");
    exit(0);
  }
  else
  {
    printf("training files are: ");
    for(vector<string>::const_iterator p=inputfiles.begin(); p!=inputfiles.end(); ++p)
    {
      cout<<*p<<" ";
    }
    printf("\n");
  }

  Corpus corp(inputfiles);
  URRP urrp(&corp, K, alpha, beta, lambda, max_iter, burn_in, sample_lag);
  urrp.train();

  return 0;
}

void parse_parameters(const int argc, char* const* argv, vector<string> &inputfiles,
    double &alpha, double &beta, double &lambda, int &K)
{

  static struct option long_options[] =
  {
    {"inputfiles", required_argument, 0, 'i'},
    {"factor-num", required_argument, 0, 'k'},
    {"alpha", required_argument, 0, 'a'},
    {"beta", required_argument, 0, 'b'},
    {"lambda", required_argument, 0, 'l'},
  };

  /* getopt_long stores the option index here. */
  int option_index = 0;
  int c = 0;

  while(c != -1)
  {
    c = getopt_long(argc, argv, "i:k:a:b:l:", long_options, &option_index);

    switch(c)
    {
      case 'i':
        inputfiles.push_back(optarg);
        while(optind<argc) {
          if(argv[optind][0]!='-')
            inputfiles.push_back(argv[optind++]);
          else
            break;
        }
        break;

      case 'k':
        K = atoi(optarg);
        break;

      case 'a':
        alpha = atof(optarg);
        break;

      case 'b':
        beta = atof(optarg);
        break;

      case 'l':
        lambda = atof(optarg);
        break;

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        break;
    }
  }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
  {
    printf("non-option ARGV-elements: ");
    while(optind < argc)
      printf("%s ", argv[optind++]);
    putchar('\n');
  }
}

