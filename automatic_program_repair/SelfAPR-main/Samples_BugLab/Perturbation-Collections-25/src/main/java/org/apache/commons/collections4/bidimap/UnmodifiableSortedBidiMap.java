[BugLab_Wrong_Operator]^if  ( map  >=  Unmodifiable )  {^58^^^^^57^64^if  ( map instanceof Unmodifiable )  {^[CLASS] UnmodifiableSortedBidiMap  [METHOD] unmodifiableSortedBidiMap [RETURN_TYPE] <K,V>   SortedBidiMap<K, ? extends V> map [VARIABLES] UnmodifiableSortedBidiMap  inverse  SortedBidiMap  map  tmpMap  boolean  
[BugLab_Variable_Misuse]^return new UnmodifiableSortedBidiMap<K, V> ( this ) ;^63^^^^^57^64^return new UnmodifiableSortedBidiMap<K, V> ( map ) ;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] unmodifiableSortedBidiMap [RETURN_TYPE] <K,V>   SortedBidiMap<K, ? extends V> map [VARIABLES] UnmodifiableSortedBidiMap  inverse  SortedBidiMap  map  tmpMap  boolean  
[BugLab_Variable_Misuse]^return UnmodifiableSet.unmodifiableSet ( null ) ;^108^^^^^106^109^return UnmodifiableSet.unmodifiableSet ( set ) ;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] keySet [RETURN_TYPE] Set   [VARIABLES] UnmodifiableSortedBidiMap  inverse  Set  set  boolean  
[BugLab_Wrong_Operator]^if  ( inverse != null )  {^133^^^^^132^138^if  ( inverse == null )  {^[CLASS] UnmodifiableSortedBidiMap  [METHOD] inverseBidiMap [RETURN_TYPE] SortedBidiMap   [VARIABLES] UnmodifiableSortedBidiMap  inverse  boolean  
[BugLab_Variable_Misuse]^inverse.2 = this;^135^^^^^132^138^inverse.inverse = this;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] inverseBidiMap [RETURN_TYPE] SortedBidiMap   [VARIABLES] UnmodifiableSortedBidiMap  inverse  boolean  
[BugLab_Variable_Misuse]^return 1;^137^^^^^132^138^return inverse;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] inverseBidiMap [RETURN_TYPE] SortedBidiMap   [VARIABLES] UnmodifiableSortedBidiMap  inverse  boolean  
[BugLab_Variable_Misuse]^final SortedMap<K, V> sm = decorated (  ) .subMap ( toKey, toKey ) ;^142^^^^^141^144^final SortedMap<K, V> sm = decorated (  ) .subMap ( fromKey, toKey ) ;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] UnmodifiableSortedBidiMap  inverse  K  fromKey  toKey  boolean  SortedMap  sm  
[BugLab_Variable_Misuse]^final SortedMap<K, V> sm = decorated (  ) .subMap ( fromKey, fromKey ) ;^142^^^^^141^144^final SortedMap<K, V> sm = decorated (  ) .subMap ( fromKey, toKey ) ;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] UnmodifiableSortedBidiMap  inverse  K  fromKey  toKey  boolean  SortedMap  sm  
[BugLab_Argument_Swapping]^final SortedMap<K, V> sm = decorated (  ) .subMap ( toKey, fromKey ) ;^142^^^^^141^144^final SortedMap<K, V> sm = decorated (  ) .subMap ( fromKey, toKey ) ;^[CLASS] UnmodifiableSortedBidiMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] UnmodifiableSortedBidiMap  inverse  K  fromKey  toKey  boolean  SortedMap  sm  
