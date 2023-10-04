#include "publisher.h"

namespace pc::publisher {

std::vector<Publisher> _instances;

void add(Publisher publisher) { _instances.push_back(publisher); }

void remove(Publisher publisher) {
  auto it = std::remove(_instances.begin(), _instances.end(), publisher);
  if (it != _instances.end()) {
    _instances.erase(it, _instances.end());
  }
}

} // namespace pc::publisher
