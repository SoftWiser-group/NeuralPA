[P1_Replace_Type]^private static final  int  serialVersionUID = 3126019624511683653L;^62^^^^^57^67^private static final long serialVersionUID = 3126019624511683653L;^[CLASS] FixedSizeSortedMap   [VARIABLES] 
[P8_Replace_Mix]^private static final  short  serialVersionUID = 3126019624511683653;^62^^^^^57^67^private static final long serialVersionUID = 3126019624511683653L;^[CLASS] FixedSizeSortedMap   [VARIABLES] 
[P5_Replace_Variable]^super ( null ) ;^86^^^^^85^87^super ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] <init> [RETURN_TYPE] SortedMap)   SortedMap<K, V> map [VARIABLES] long  serialVersionUID  SortedMap  map  boolean  
[P14_Delete_Statement]^^86^^^^^85^87^super ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] <init> [RETURN_TYPE] SortedMap)   SortedMap<K, V> map [VARIABLES] long  serialVersionUID  SortedMap  map  boolean  
[P4_Replace_Constructor]^return return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  )  ;^75^^^^^74^76^return new FixedSizeSortedMap<K, V> ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] fixedSizeSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map [VARIABLES] long  serialVersionUID  SortedMap  map  boolean  
[P8_Replace_Mix]^return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  )  ;^75^^^^^74^76^return new FixedSizeSortedMap<K, V> ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] fixedSizeSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map [VARIABLES] long  serialVersionUID  SortedMap  map  boolean  
[P7_Replace_Invocation]^out.writeObject (  ) ;^103^^^^^102^105^out.defaultWriteObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^out .writeObject ( out )  ;^103^^^^^102^105^out.defaultWriteObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^103^^^^^102^105^out.defaultWriteObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^in.defaultReadObject (  ) ;out.defaultWriteObject (  ) ;^103^^^^^102^105^out.defaultWriteObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^out.writeObject ( map ) ;out.defaultWriteObject (  ) ;^103^^^^^102^105^out.defaultWriteObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^104^^^^^102^105^out.writeObject ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^out.defaultWriteObject (  ) ;out.writeObject ( map ) ;^104^^^^^102^105^out.writeObject ( map ) ;^[CLASS] FixedSizeSortedMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P8_Replace_Mix]^in .readObject ( in )  ;^112^^^^^111^114^in.defaultReadObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P14_Delete_Statement]^^112^^^^^111^114^in.defaultReadObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^out.defaultWriteObject (  ) ;in.defaultReadObject (  ) ;^112^^^^^111^114^in.defaultReadObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P8_Replace_Mix]^map =   ( Map<K, V> )  null.readObject (  ) ;^113^^^^^111^114^map =  ( Map<K, V> )  in.readObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P14_Delete_Statement]^^113^^^^^111^114^map =  ( Map<K, V> )  in.readObject (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P2_Replace_Operator]^if  ( map.containsKey ( key )  > false )  {^119^^^^^118^123^if  ( map.containsKey ( key )  == false )  {^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P3_Replace_Literal]^if  ( map.containsKey ( key )  == true )  {^119^^^^^118^123^if  ( map.containsKey ( key )  == false )  {^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^if  ( key.containsKey ( map )  == false )  {^119^^^^^118^123^if  ( map.containsKey ( key )  == false )  {^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P7_Replace_Invocation]^if  ( map.putAll ( key )  == false )  {^119^^^^^118^123^if  ( map.containsKey ( key )  == false )  {^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P7_Replace_Invocation]^if  ( map .entrySet (  )   == false )  {^119^^^^^118^123^if  ( map.containsKey ( key )  == false )  {^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Cannot put new key/value pair - Map is fixed size");^119^120^121^^^118^123^if  ( map.containsKey ( key )  == false )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P16_Remove_Block]^^119^120^121^^^118^123^if  ( map.containsKey ( key )  == false )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P13_Insert_Block]^if  ( isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  {     throw new IllegalArgumentException ( "Cannot put new key/value pair - Map is fixed size" ) ; }^119^^^^^118^123^[Delete]^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P13_Insert_Block]^if  (  ( map.containsKey ( key )  )  == false )  {     throw new IllegalArgumentException ( "Cannot put new key/value pair - Map is fixed size" ) ; }^120^^^^^118^123^[Delete]^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new IllegalArgumentException  (" ")  ;^120^^^^^118^123^throw new IllegalArgumentException  (" ")  ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put (  value ) ;^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put ( key ) ;^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return key.put ( map, value ) ;^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put ( value, key ) ;^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return value.put ( key, map ) ;^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P14_Delete_Statement]^^122^^^^^118^123^return map.put ( key, value ) ;^[CLASS] FixedSizeSortedMap  [METHOD] put [RETURN_TYPE] V   final K key final V value [VARIABLES] K  key  boolean  long  serialVersionUID  V  value  
[P5_Replace_Variable]^if  ( CollectionUtils.isSubCollection ( 1.keySet (  ) , keySet (  )  )  )  {^127^^^^^126^131^if  ( CollectionUtils.isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  {^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^if  ( CollectionUtils.isSubCollection ( mapToCopy .keySet (  )  , keySet (  )  )  )  {^127^^^^^126^131^if  ( CollectionUtils.isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  {^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^if  ( CollectionUtils.isSubCollection ( mapToCopy.size (  ) , keySet (  )  )  )  {^127^^^^^126^131^if  ( CollectionUtils.isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  {^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Cannot put new key/value pair - Map is fixed size");^127^128^129^^^126^131^if  ( CollectionUtils.isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P16_Remove_Block]^^127^128^129^^^126^131^if  ( CollectionUtils.isSubCollection ( mapToCopy.keySet (  ) , keySet (  )  )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P13_Insert_Block]^if  (  ( map.containsKey ( key )  )  == false )  {     throw new IllegalArgumentException ( "Cannot put new key/value pair - Map is fixed size" ) ; }^127^^^^^126^131^[Delete]^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P13_Insert_Block]^if  (  ( map.containsKey ( key )  )  == false )  {     throw new IllegalArgumentException ( "Cannot put new key/value pair - Map is fixed size" ) ; }^128^^^^^126^131^[Delete]^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^throw new UnsupportedOperationException  (" ")  ;throw new IllegalArgumentException  (" ")  ;^128^^^^^126^131^throw new IllegalArgumentException  (" ")  ;^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P5_Replace_Variable]^map.putAll ( map ) ;^130^^^^^126^131^map.putAll ( mapToCopy ) ;^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P5_Replace_Variable]^map.putAll ( mapToCopyToCopy ) ;^130^^^^^126^131^map.putAll ( mapToCopy ) ;^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^map.containsKey ( mapToCopy ) ;^130^^^^^126^131^map.putAll ( mapToCopy ) ;^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^130^^^^^126^131^map.putAll ( mapToCopy ) ;^[CLASS] FixedSizeSortedMap  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ? extends V> mapToCopy [VARIABLES] Map  mapToCopy  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^135^^^^^134^136^throw new UnsupportedOperationException  (" ")  ;^[CLASS] FixedSizeSortedMap  [METHOD] clear [RETURN_TYPE] void   [VARIABLES] long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^throw new IllegalArgumentException  (" ")  ;throw new UnsupportedOperationException  (" ")  ;^140^^^^^139^141^throw new UnsupportedOperationException  (" ")  ;^[CLASS] FixedSizeSortedMap  [METHOD] remove [RETURN_TYPE] V   Object key [VARIABLES] long  serialVersionUID  Object  key  boolean  
[P7_Replace_Invocation]^return UnmodifiableSet.unmodifiableSet ( map.keySet (  )  ) ;^145^^^^^144^146^return UnmodifiableSet.unmodifiableSet ( map.entrySet (  )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] entrySet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^145^146^^^^144^146^return UnmodifiableSet.unmodifiableSet ( map.entrySet (  )  ) ; }^[CLASS] FixedSizeSortedMap  [METHOD] entrySet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return UnmodifiableSet.unmodifiableSet ( map.entrySet (  )  ) ;^150^^^^^149^151^return UnmodifiableSet.unmodifiableSet ( map.keySet (  )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] keySet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return UnmodifiableSet.unmodifiableSet ( map .keySet (  )   ) ;^150^^^^^149^151^return UnmodifiableSet.unmodifiableSet ( map.keySet (  )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] keySet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^150^151^^^^149^151^return UnmodifiableSet.unmodifiableSet ( map.keySet (  )  ) ; }^[CLASS] FixedSizeSortedMap  [METHOD] keySet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return UnmodifiableCollection.unmodifiableCollection ( map.keySet (  )  ) ;^155^^^^^154^156^return UnmodifiableCollection.unmodifiableCollection ( map.values (  )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] values [RETURN_TYPE] Collection   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^155^156^^^^154^156^return UnmodifiableCollection.unmodifiableCollection ( map.values (  )  ) ; }^[CLASS] FixedSizeSortedMap  [METHOD] values [RETURN_TYPE] Collection   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return UnmodifiableCollection.unmodifiableCollection ( map .containsKey ( 1 )   ) ;^155^^^^^154^156^return UnmodifiableCollection.unmodifiableCollection ( map.values (  )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] values [RETURN_TYPE] Collection   [VARIABLES] long  serialVersionUID  boolean  
[P4_Replace_Constructor]^return return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  )  .subMap ( fromKey, toKey )  ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P4_Replace_Constructor]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap (  toKey )  ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P4_Replace_Constructor]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey )  ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P5_Replace_Variable]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( toKey, fromKey )  ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( keySet (  ) .subMap ( fromKey, toKey )  ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  )  .headMap ( toKey )   ) ;^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P14_Delete_Statement]^^161^^^^^160^162^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] long  serialVersionUID  K  fromKey  toKey  boolean  
[P4_Replace_Constructor]^return return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  )  .headMap ( toKey )  ) ;^166^^^^^165^167^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] long  serialVersionUID  K  toKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( toKey )  ) ;^166^^^^^165^167^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] long  serialVersionUID  K  toKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( keySet (  ) .headMap ( toKey )  ) ;^166^^^^^165^167^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] long  serialVersionUID  K  toKey  boolean  
[P14_Delete_Statement]^^166^^^^^165^167^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( toKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] long  serialVersionUID  K  toKey  boolean  
[P4_Replace_Constructor]^return return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  )  .tailMap ( fromKey )  ) ;^171^^^^^170^172^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( fromKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] long  serialVersionUID  K  fromKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .headMap ( fromKey )  ) ;^171^^^^^170^172^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( fromKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] long  serialVersionUID  K  fromKey  boolean  
[P7_Replace_Invocation]^return new FixedSizeSortedMap<K, V> ( keySet (  ) .tailMap ( fromKey )  ) ;^171^^^^^170^172^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( fromKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] long  serialVersionUID  K  fromKey  boolean  
[P8_Replace_Mix]^return  new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .subMap ( fromKey, toKey )  )  .headMap ( fromKey )  ) ;^171^^^^^170^172^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( fromKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] long  serialVersionUID  K  fromKey  boolean  
[P14_Delete_Statement]^^171^^^^^170^172^return new FixedSizeSortedMap<K, V> ( getSortedMap (  ) .tailMap ( fromKey )  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] long  serialVersionUID  K  fromKey  boolean  
[P3_Replace_Literal]^return false;^175^^^^^174^176^return true;^[CLASS] FixedSizeSortedMap  [METHOD] isFull [RETURN_TYPE] boolean   [VARIABLES] long  serialVersionUID  boolean  
[P3_Replace_Literal]^return size() + 4 ;^179^^^^^178^180^return size (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] maxSize [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return keySet (  ) ;^179^^^^^178^180^return size (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] maxSize [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P3_Replace_Literal]^return size() - 6 ;^179^^^^^178^180^return size (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] maxSize [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^179^^^^^178^180^return size (  ) ;^[CLASS] FixedSizeSortedMap  [METHOD] maxSize [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  