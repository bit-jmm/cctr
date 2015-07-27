#pragma once

#include <vector>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <string>
#include <cctype>
#include <iostream>
#include <fstream>
#include "omp.h"
#include "map"
#include "set"
#include "algorithm"
#include "sstream"
#include "gzstream.h"
#include "../../libs/json.hpp"
/// Data associated with a rating
struct rating
{
  int user; // ID of the user
  int item; // ID of the item
  int value; // Rating

  int attitude=0;

  int ratingTime; // Unix-time of the rating
  std::vector< std::pair<int, int> > words; // IDs of the words in the review
};

typedef struct rating rating;
/// Safely open a file
FILE* fopen_(const char* p, const char* m);
std::string filename_in_path(std::string &path);

bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2);
/// To sort ratings by product ID
bool ratingCompare(rating* v1, rating* v2);
template<typename T> int sgn(T val);
void array_init(double *array, int size, double value);

class Corpus
{
public:
  std::vector<rating*>* V;

  int nUsers; // Number of users
  int nItems; // Number of items
  int nWords; // Number of words

  std::map<std::string, int> user2id; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> item2id; // Maps an item's string-valued ID to an integer

  std::map<int, std::string> id2user; // Inverse of the above map
  std::map<int, std::string> id2item;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> word2id; // Map each word to its integer ID
  std::map<int, std::string> id2word; // Inverse of the above map

  std::string input_filename;

  Corpus(std::vector<std::string> ratingFiles)
  {
    std::map<std::string, int> uCounts;
    std::map<std::string, int> bCounts;

    std::string uName;
    std::string bName;
    float value;
    int ratingTime;
    int nw;
    int nRead = 0;

    std::string sWord;

    // Read the input files. The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary
    //stop words
    std::map<std::string, int> stop_words;
    std::ifstream stop_file;
    stop_file.open("../../data/text_preprocess/en.json", std::ios::in);
    nlohmann::json j;
    j << stop_file;
    for(std::string word : j)
    {
      stop_words[word] = 1;
    }
    stop_file.close();

    std::vector<std::string> lines;
    std::string line;
    for (std::vector<std::string>::iterator it=ratingFiles.begin(); it!=ratingFiles.end();++it)
    {
      lines = readLines(*it);
      if(input_filename.length()==0) {
        input_filename = filename_in_path(*it);
      } else {
        input_filename += "_" + filename_in_path(*it);
      }
      for(std::vector<std::string>::iterator it=lines.begin(); it!=lines.end();++it)
      {
        line = *it;
        std::stringstream ss(line);
        ss >> uName >> bName >> value >> ratingTime >> nw;
        if (value > 5 or value < 0)
        { // Ratings should be in the range [0,5]
          printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), ratingTime);
          continue;
        }

        // item number up to 5000
        if (bCounts.size()==5000 && bCounts.find(bName)==bCounts.end()) {
          continue;
        }
        for (int w = 0; w < nw; w++)
        {
          ss >> sWord;
          if(stop_words.find(sWord) != stop_words.end())
            continue;
          if (wordCount.find(sWord) == wordCount.end())
            wordCount[sWord] = 0;
          wordCount[sWord]++;
        }

        if (uCounts.find(uName) == uCounts.end())
          uCounts[uName] = 0;
        if (bCounts.find(bName) == bCounts.end())
          bCounts[bName] = 0;
        uCounts[uName]++;
        bCounts[bName]++;

        nRead++;
        if (nRead % 100000 == 0)
        {
          printf(".");
          fflush(stdout);
        }
      }
    }

    V = new std::vector<rating*>();
    rating* v = new rating();
    std::map<std::string, int> user2id;
    std::map<std::string, int> item2id;

    nUsers = 0;
    nItems = 0;
    // Comment this block to include all users, otherwise only users/items with userMin/itemMin ratings will be considered
    //    nUsers = 1;
    //    nItems = 1;
    //    user2id["NOT_ENOUGH_ratingS"] = 0;
    //    item2id["NOT_ENOUGH_ratingS"] = 0;
    //    id2user[0] = "NOT_ENOUGH_ratingS";
    //    id2item[0] = "NOT_ENOUGH_ratingS";
    //    rating* v_ = new rating();
    //    v_->user = 0;
    //    v_->item = 0;
    //    v_->value = 0;
    //    v_->ratingTime = 0;
    //    V->push_back(v_);

    int userMin = 0;
    int itemMin = 0;

    int maxWords = 3000; // Dictionary size
    std::vector < std::pair<std::string, int> > whichWords;
    for (std::map<std::string, int>::iterator it = wordCount.begin(); it != wordCount.end(); it++)
      whichWords.push_back(*it);
    sort(whichWords.begin(), whichWords.end(), wordCountCompare);
    if ((int) whichWords.size() < maxWords)
      maxWords = (int) whichWords.size();
    nWords = maxWords;
    for (int w = 0; w < maxWords; w++)
    {
      word2id[whichWords[w].first] = w;
      id2word[w] = whichWords[w].first;
    }

    int all_words_count = 0;
    // Re-read the entire file, this time building structures from those words in the dictionary
    for (std::vector<std::string>::iterator it=ratingFiles.begin(); it!=ratingFiles.end();++it)
    {
      lines = readLines(*it);
      for(std::vector<std::string>::iterator it=lines.begin(); it!=lines.end();++it)
      {
        line = *it;
        std::stringstream ss(line);
        ss >> uName >> bName >> value >> ratingTime >> nw;

        if(bCounts.find(bName) == bCounts.end()) {
          continue;
        }

        for (int w = 0; w < nw; w++)
        {
          ss >> sWord;
          if (word2id.find(sWord) != word2id.end())
          {
            all_words_count++;
            v->words.push_back(std::make_pair(word2id[sWord], 0));
          }
        }

        if (uCounts[uName] >= userMin)
        {
          if (user2id.find(uName) == user2id.end())
          {
            id2user[nUsers] = uName;
            user2id[uName] = nUsers++;
          }
          v->user = user2id[uName];
        }
        else
          v->user = 0;

        if (bCounts[bName] >= itemMin)
        {
          if (item2id.find(bName) == item2id.end())
          {
            id2item[nItems] = bName;
            item2id[bName] = nItems++;
          }
          v->item = item2id[bName];
        }
        else
          v->item = 0;

        v->value = int(value)-1;
        v->ratingTime = ratingTime;

        V->push_back(v);
        v = new rating();
      }
    }

    printf("\nuser count: %d, item count: %d, review count: %zu, all_words_count: %d\n", nUsers, nItems, V->size(), all_words_count);
    delete v;
  }

  std::vector<std::string> readLines(std::string filename);

  ~Corpus()
  {
    for (std::vector<rating*>::iterator it = V->begin(); it != V->end(); it++)
      delete *it;
    delete V;
  }

};
