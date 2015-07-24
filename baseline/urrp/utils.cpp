#include "utils.h"

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
  size_t begin = 0;
  size_t end = path.length();
  size_t pos = path.find_last_of("/\\");
  if (pos != std::string::npos) {
    begin = pos;
  }
  pos = path.find_last_of(".");
  if (pos != std::string::npos) {
    end = pos;
  }
  return path.substr(begin+1, end-begin-1);
}

bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort ratings by product ID
bool ratingCompare(rating* v1, rating* v2)
{
  return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

void array_init(double *array, int size, double value)
{
  for(int i=0; i<size; i++)
  {
    *(array+i) = value;
  }
}

std::vector<std::string> Corpus::readLines(std::string filename)
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
      }
      in.close();
    }
    return lines;
}
