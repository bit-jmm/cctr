#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include "utils.h"
#include "../../libs/json.hpp"
#include <boost/algorithm/string.hpp>
using json = nlohmann::json;
using namespace std;
namespace fs = boost::filesystem;
namespace ba = boost::algorithm;

gsl_rng * RANDOM_NUMBER = NULL;

void json2txt(std::string source_file, std::string dest_file)
{
  igzstream in;
  in.open(source_file.c_str());
  ogzstream outfile(dest_file.c_str());
  string line;

  map<string, int> uCount;
  map<string, int> iCount;
  long wordCount = 0;

  int nRead = 0;
  while (std::getline(in, line))
  {
    auto j = json::parse(line);
    string uid = j["reviewerID"];
    if(uCount.find(uid) == uCount.end())
      uCount[uid] = 1;

    string iid = j["asin"];
    if(iCount.find(iid) == iCount.end())
      iCount[iid] = 1;

    float rating = j["overall"];
    long rtime = j["unixReviewTime"];
    string reviewText = j["reviewText"];

    vector<string> * words = tokenizer(reviewText);
    string parsed_review_text = "";
    for(vector<string>::iterator it=words->begin(); it!=words->end(); ++it)
    {
      parsed_review_text += *it + " ";
    }
    wordCount += words->size();
    ba::trim_right(parsed_review_text);

    ostringstream out;
    out << uid << " " << iid << " " << rating << " " << rtime << " " << words->size() << " " << parsed_review_text;
    line = out.str();
    outfile << line << endl;
    nRead++;
    if (nRead % 100000 == 0)
    {
      printf(".");
      fflush(stdout);
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

  fs::path dest_bp(dest_path);

  vector<fs::path> *files = files_in_path((const char *)source_path);

  for(vector<fs::path>::const_iterator it = files->begin(); it != files->end(); it++)
  {
    //concat dest_path
    string dfilename = it->filename().string();
    dfilename.replace(dfilename.find("json"), 4, "txt");
    fs::path dfile(dest_bp);
    fs::path fn(dfilename);
    dfile /= fn;
    cout << *it << " : " << dfile << endl;
    json2txt(it->string(), dfile.string());
  }
  return 0;
}
