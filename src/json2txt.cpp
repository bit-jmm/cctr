#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <omp.h>
#include "utils.h"
#include "../../libs/json.hpp"
using json = nlohmann::json;
using namespace std;
#define NUM_THREADS 10
gsl_rng * RANDOM_NUMBER = NULL;

void json2txt(string source_file, string dest_file)
{
  igzstream in;
  in.open(source_file.c_str());
  ogzstream outfile(dest_file.c_str());
  string line;

  map<string, int> uCount;
  map<string, int> iCount;
  long wordCount = 0;
  vector<string> lines;

  while (getline(in, line))
  {
    lines.push_back(line);
  }

  omp_set_num_threads(NUM_THREADS);
  int nRead = 0, i;
  int rating_num = lines.size();
#pragma omp parallel for
  for (i=0; i<rating_num; i++)
  {
    auto j = json::parse(lines[i]);

    if(j["reviewerID"].is_null() || j["asin"].is_null() ||
        j["overall"].is_null() || j["unixReviewTime"].is_null() ||
        j["reviewText"].is_null())
    {
      continue;
    }
    string uid = j["reviewerID"];
    string iid = j["asin"];

    float rating = j["overall"];
    long rtime = j["unixReviewTime"];
    string reviewText = j["reviewText"];
#pragma omp critical
    {
      if(uCount.find(uid) == uCount.end())
        uCount[uid] = 1;

      if(iCount.find(iid) == iCount.end())
        iCount[iid] = 1;
    }
    vector<string> * words = tokenizer(reviewText);
    string parsed_review_text = "";
    for(vector<string>::iterator it=words->begin(); it!=words->end(); ++it)
    {
      parsed_review_text += *it + " ";
    }
#pragma omp critical
    {
      wordCount += words->size();
    }

    ostringstream out;
    out << uid << " " << iid << " " << rating << " " << rtime << " " << words->size() << " " << parsed_review_text;
    line = out.str();
#pragma omp critical
    {
      outfile << line << endl;
      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush(stdout);
      }
    }
    delete words;
  }
  cout << endl << "user_num : " << uCount.size() << endl;
  cout << "item_num : " << iCount.size() << endl;
  cout << "review_num : " << nRead << endl;
  cout << "word_num : " << wordCount << endl << endl;

  outfile.close();
  in.close();
}

int main(int argc, char* argv[])
{
  time_t t; time(&t);
  long   random_seed = (long) t;
  RANDOM_NUMBER = new_random_number_generator(random_seed);

  if (argc < 3) {
    printf("\nPlease input source folder and destination folder.\n");
    exit(0);
  }

  const char * source_path = argv[1];
  const char * dest_path = argv[2];

  if(!dir_exists(dest_path))
  {
    make_directory(dest_path);
  }

  vector<string> *files = files_in_path((const char *)source_path);

  for(vector<string>::const_iterator it = files->begin(); it != files->end(); it++)
  {
    string dfilename = *it;
    dfilename.replace(dfilename.find("json"), 4, "txt");
    string dfilepath;
    string sfilepath;

    if(source_path[strlen(source_path)-1] == '/')
    {
      sfilepath = string(source_path) + *it;
    }
    else
    {
      sfilepath = string(source_path) + "/" + *it;
    }

    if(dest_path[strlen(dest_path)-1] == '/')
    {
      dfilepath = string(dest_path) + dfilename;
    }
    else
    {
      dfilepath = string(dest_path) + "/" + dfilename;
    }
    cout << sfilepath << " : " << dfilepath << endl;
    json2txt(sfilepath, dfilepath);
  }

  return 0;
}
