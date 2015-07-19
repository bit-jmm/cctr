#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <cctype>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include "vector"
#include "common.hpp"
#include "algorithm"
#include "lbfgs.h"
#include "sstream"
#include "gzstream.h"
#include "../../libs/json.hpp"

// for convenience
using json = nlohmann::json;

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
  FILE* f = fopen(p, m);
  if (!f)
  {
    printf("Failed to open %s\n", p);
    exit(1);
  }
  return f;
}

std::string filename_in_path(std::string &path)
{
  int begin = 0;
  int end = path.length();
  int pos = path.find_last_of("/\\");
  if (pos != std::string::npos) {
    begin = pos;
  }
  pos = path.find_last_of(".");
  if (pos != std::string::npos) {
    end = pos;
  }
  return path.substr(begin+1, end-begin-1);
}

/// Data associated with a rating
struct vote
{
  int user; // ID of the user
  int item; // ID of the item
  float value; // Rating

  int voteTime; // Unix-time of the rating
  std::vector<int> words; // IDs of the words in the review
};

typedef struct vote vote;

/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(vote* v1, vote* v2)
{
  return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

class corpus
{
public:
  std::vector<vote*>* V;

  int nUsers; // Number of users
  int nBeers; // Number of items
  int nWords; // Number of words

  std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> beerIds; // Maps an item's string-valued ID to an integer

  std::map<int, std::string> rUserIds; // Inverse of the above map
  std::map<int, std::string> rBeerIds;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> wordId; // Map each word to its integer ID
  std::map<int, std::string> idWord; // Inverse of the above map

  std::string input_filename;

  corpus(std::vector<std::string> voteFiles, int max)
  {
    std::map<std::string, int> uCounts;
    std::map<std::string, int> bCounts;

    std::string uName;
    std::string bName;
    float value;
    int voteTime;
    int nw;
    int nRead = 0;

    std::string sWord;

    // Read the input files. The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary

    std::vector<std::string> lines;
    std::string line;
    for (std::vector<std::string>::iterator it=voteFiles.begin(); it!=voteFiles.end();++it)
    {
      lines = readLines(*it, max);
      if(input_filename.length()==0) {
        input_filename = filename_in_path(*it);
      } else {
        input_filename += "_" + filename_in_path(*it);
      }
      for(std::vector<std::string>::iterator it=lines.begin(); it!=lines.end();++it)
      {
        line = *it;
        std::stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime >> nw;
        if (value > 5 or value < 0)
        { // Ratings should be in the range [0,5]
          printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), voteTime);
          exit(0);
        }

        // item number up to 5000
        if (bCounts.size()==5000 && bCounts.find(bName)==bCounts.end()) {
          continue;
        }
        for (int w = 0; w < nw; w++)
        {
          ss >> sWord;
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

        if (max > 0 and (int) nRead >= max)
          break;
      }
    }

    printf("\nnUsers = %d, nItems = %d, nRatings = %d\n", (int) uCounts.size(), (int) bCounts.size(), nRead);

    V = new std::vector<vote*>();
    vote* v = new vote();
    std::map<std::string, int> userIds;
    std::map<std::string, int> beerIds;

    nUsers = 0;
    nBeers = 0;
    // Comment this block to include all users, otherwise only users/items with userMin/beerMin ratings will be considered
    //    nUsers = 1;
    //    nBeers = 1;
    //    userIds["NOT_ENOUGH_VOTES"] = 0;
    //    beerIds["NOT_ENOUGH_VOTES"] = 0;
    //    rUserIds[0] = "NOT_ENOUGH_VOTES";
    //    rBeerIds[0] = "NOT_ENOUGH_VOTES";
    //    vote* v_ = new vote();
    //    v_->user = 0;
    //    v_->item = 0;
    //    v_->value = 0;
    //    v_->voteTime = 0;
    //    V->push_back(v_);

    int userMin = 0;
    int beerMin = 0;

    int maxWords = 5000; // Dictionary size
    std::vector < std::pair<std::string, int> > whichWords;
    for (std::map<std::string, int>::iterator it = wordCount.begin(); it != wordCount.end(); it++)
      whichWords.push_back(*it);
    sort(whichWords.begin(), whichWords.end(), wordCountCompare);
    if ((int) whichWords.size() < maxWords)
      maxWords = (int) whichWords.size();
    nWords = maxWords;
    for (int w = 0; w < maxWords; w++)
    {
      wordId[whichWords[w].first] = w;
      idWord[w] = whichWords[w].first;
    }

    // Re-read the entire file, this time building structures from those words in the dictionary
    for (std::vector<std::string>::iterator it=voteFiles.begin(); it!=voteFiles.end();++it)
    {
      lines = readLines(*it, max);
      for(std::vector<std::string>::iterator it=lines.begin(); it!=lines.end();++it)
      {
        line = *it;
        std::stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime >> nw;

        if(bCounts.find(bName) == bCounts.end()) {
          continue;
        }

        for (int w = 0; w < nw; w++)
        {
          ss >> sWord;
          if (wordId.find(sWord) != wordId.end())
            v->words.push_back(wordId[sWord]);
        }

        if (uCounts[uName] >= userMin)
        {
          if (userIds.find(uName) == userIds.end())
          {
            rUserIds[nUsers] = uName;
            userIds[uName] = nUsers++;
          }
          v->user = userIds[uName];
        }
        else
          v->user = 0;

        if (bCounts[bName] >= beerMin)
        {
          if (beerIds.find(bName) == beerIds.end())
          {
            rBeerIds[nBeers] = bName;
            beerIds[bName] = nBeers++;
          }
          v->item = beerIds[bName];
        }
        else
          v->item = 0;

        v->value = value;
        v->voteTime = voteTime;

        V->push_back(v);
        v = new vote();
      }
    }

    printf("user count: %d, item count: %d, review count: %d\n", nUsers, nBeers, V->size());
    delete v;
  }

  std::vector<std::string> readLines(std::string filename, int max)
  {
      std::vector<std::string> lines;
      int nRead = 0;
      igzstream in;
      in.open(filename.c_str());
      std::string line;
      if (filename.find("json") == std::string::npos)
      {
        while (std::getline(in, line))
        {
          lines.push_back(line);
          nRead++;
          if (nRead % 100000 == 0)
          {
            printf(".");
            fflush(stdout);
          }

          if (max > 0 and (int) nRead >= max)
            break;
        }
        in.close();
      }
      else
      {
        while (std::getline(in, line))
        {
          auto j = json::parse(line);
          std::string uid = j["reviewerID"];
          std::string iid = j["asin"];
          float rating = j["overall"];
          long rtime = j["unixReviewTime"];
          std::string reviewText = j["reviewText"];

          const char* str = reviewText.c_str();
          int word_count = 0; // Holds number of words
          for(int i = 0; str[i] != '\0'; i++)
          {
            if (isspace(str[i])) //Checking for spaces
            {
              word_count++;
              i++;
              while(str[i] != '\0' && isspace(str[i])) i++;
              i--;
            }
          }

          line = uid + " " + iid + " " + std::to_string(rating) + " " +
            std::to_string(rtime) + " " + std::to_string(word_count) + " " + reviewText;
          lines.push_back(line);

          nRead++;
          if (nRead % 100000 == 0)
          {
            printf(".");
            fflush(stdout);
          }

          if (max > 0 and (int) nRead >= max)
            break;
        }
        in.close();
      }
      return lines;
  }

  ~corpus()
  {
    for (std::vector<vote*>::iterator it = V->begin(); it != V->end(); it++)
      delete *it;
    delete V;
  }

};
