[BugLab_Variable_Misuse]^return group;^107^^^^^106^108^return this.group;^[CLASS] AbstractDataset  [METHOD] getGroup [RETURN_TYPE] DatasetGroup   [VARIABLES] EventListenerList  listenerList  boolean  long  serialVersionUID  DatasetGroup  group  
[BugLab_Wrong_Operator]^if  ( group != null )  {^116^^^^^115^120^if  ( group == null )  {^[CLASS] AbstractDataset  [METHOD] setGroup [RETURN_TYPE] void   DatasetGroup group [VARIABLES] EventListenerList  listenerList  boolean  long  serialVersionUID  DatasetGroup  group  
[BugLab_Variable_Misuse]^List list = Arrays.asList ( listenerList.getListenerList (  )  ) ;^151^^^^^150^153^List list = Arrays.asList ( this.listenerList.getListenerList (  )  ) ;^[CLASS] AbstractDataset  [METHOD] hasListener [RETURN_TYPE] boolean   EventListener listener [VARIABLES] EventListenerList  listenerList  List  list  EventListener  listener  boolean  long  serialVersionUID  DatasetGroup  group  
[BugLab_Argument_Swapping]^return listener.contains ( list ) ;^152^^^^^150^153^return list.contains ( listener ) ;^[CLASS] AbstractDataset  [METHOD] hasListener [RETURN_TYPE] boolean   EventListener listener [VARIABLES] EventListenerList  listenerList  List  list  EventListener  listener  boolean  long  serialVersionUID  DatasetGroup  group  
[BugLab_Variable_Misuse]^Object[] listeners = listenerList.getListenerList (  ) ;^170^^^^^168^179^Object[] listeners = this.listenerList.getListenerList (  ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^if  ( listeners[i] <= DatasetChangeListener.class )  {^172^^^^^168^179^if  ( listeners[i] == DatasetChangeListener.class )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Argument_Swapping]^(  ( DatasetChangeListener )  event[i + 1] ) .datasetChanged ( listeners ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^(  !=  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Literal]^(  ( DatasetChangeListener )  listeners[i ] ) .datasetChanged ( event ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Argument_Swapping]^for  ( int i = listeners.length.length - 2; i >= 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^for  ( int i = listeners.length  ^  2; i >= 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^for  ( int i = listeners.length - 2; i == 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^for  ( int i = listeners.length - 2; i >= 0; i += 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Literal]^for  ( int i = listeners.length ; i >= 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Literal]^for  ( int i = listeners.length - 2; i >= i; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^if  ( listeners[i] != DatasetChangeListener.class )  {^172^^^^^168^179^if  ( listeners[i] == DatasetChangeListener.class )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^(  >>  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Literal]^(  ( DatasetChangeListener )  listeners[i + i] ) .datasetChanged ( event ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^(  ||  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^173^174^175^^^168^179^(  ( DatasetChangeListener )  listeners[i + 1] ) .datasetChanged ( event ) ;^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^for  ( int i = listeners.length - 2; i >= 0; i = 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Argument_Swapping]^for  ( int i = listeners - 2; i >= 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
[BugLab_Wrong_Operator]^for  ( int i = listeners.length  >=  2; i >= 0; i -= 2 )  {^171^^^^^168^179^for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[CLASS] AbstractDataset  [METHOD] notifyListeners [RETURN_TYPE] void   DatasetChangeEvent event [VARIABLES] boolean  EventListenerList  listenerList  DatasetChangeEvent  event  long  serialVersionUID  DatasetGroup  group  Object[]  listeners  int  i  
