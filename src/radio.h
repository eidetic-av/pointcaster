#include <thread>

namespace bob::pointcaster {
  
  class Radio {
  private:
    std::jthread radio_thread;
  public:
    const int port;
    Radio(const int _port);
    ~Radio();
  };

}
