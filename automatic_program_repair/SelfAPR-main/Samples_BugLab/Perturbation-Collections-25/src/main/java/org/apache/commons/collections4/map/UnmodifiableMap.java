[BugLab_Wrong_Operator]^if  ( map  <  Unmodifiable )  {^63^^^^^62^69^if  ( map instanceof Unmodifiable )  {^[CLASS] UnmodifiableMap  [METHOD] unmodifiableMap [RETURN_TYPE] <K,V>   Map<? extends K, ? extends V> map [VARIABLES] Map  map  tmpMap  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^out.writeObject ( 3 ) ;^93^^^^^91^94^out.writeObject ( map ) ;^[CLASS] UnmodifiableMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( map  >=  IterableMap )  {^133^^^^^132^139^if  ( map instanceof IterableMap )  {^[CLASS] UnmodifiableMap  [METHOD] mapIterator [RETURN_TYPE] MapIterator   [VARIABLES] long  serialVersionUID  MapIterator  it  boolean  
[BugLab_Variable_Misuse]^return UnmodifiableMapIterator.unmodifiableMapIterator ( 3 ) ;^135^^^^^132^139^return UnmodifiableMapIterator.unmodifiableMapIterator ( it ) ;^[CLASS] UnmodifiableMap  [METHOD] mapIterator [RETURN_TYPE] MapIterator   [VARIABLES] long  serialVersionUID  MapIterator  it  boolean  
