#include "publisher.h"

namespace pc::publisher {

  std::vector<Publisher> _instances;

  void add(Publisher publisher) {
    _instances.push_back(publisher);
  }
}
