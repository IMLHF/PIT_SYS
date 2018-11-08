import sys


class _const(object):
  def __init__(self):
    self.RAWDATA_DIR = '/mnt/d/tf_recipe/PIT_SYS/utterance_test/speaker_set'
    # self.RAWDATA_DIR = '/home/student/work/pit_test/data_small'

  class ConstError(PermissionError):
    pass

  def __setattr__(self, name, value):
    if name in self.__dict__.keys():
      raise self.ConstError("Can't rebind const(%s)" % name)
    self.__dict__[name] = value

  def __delattr__(self, name):
    if name in self.__dict__:
      raise self.ConstError("Can't unbind const(%s)" % name)
    raise NameError(name)


sys.modules[__name__] = _const()
