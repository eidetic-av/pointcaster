namespace bob::sensors {

class Driver {
public:
  virtual bool Open() = 0;
  virtual bool Close() = 0;
  virtual bool IsOpen() = 0;
};

} // namespace bob::sensors
