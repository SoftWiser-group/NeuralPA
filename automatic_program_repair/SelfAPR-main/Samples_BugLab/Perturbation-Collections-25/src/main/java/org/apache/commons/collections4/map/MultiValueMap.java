[BugLab_Variable_Misuse]^this ( new HashMap<K, V> (  ) , new ReflectionFactory ( ArrayList.clazz )  ) ;^134^^^^^133^135^this ( new HashMap<K, V> (  ) , new ReflectionFactory ( ArrayList.class )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] <init> [RETURN_TYPE] MultiValueMap()   [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( collectionFactory != null )  {^149^^^^^146^153^if  ( collectionFactory == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] <init> [RETURN_TYPE] Factory)   Map<K, ? super C> map Factory<C> collectionFactory [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.collectionFactory = 0;^152^^^^^146^153^this.collectionFactory = collectionFactory;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] <init> [RETURN_TYPE] Factory)   Map<K, ? super C> map Factory<C> collectionFactory [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.iterator = 1.iterator (  ) ;^519^^^^^516^520^this.iterator = values.iterator (  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] <init> [RETURN_TYPE] Object)   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  long  serialVersionUID  
[BugLab_Variable_Misuse]^return MultiValueMap.<K, V, ArrayList> multiValueMap (  ( Map<K, ? super Collection> )  map, ArrayList.clazz ) ;^90^^^^^89^91^return MultiValueMap.<K, V, ArrayList> multiValueMap (  ( Map<K, ? super Collection> )  map, ArrayList.class ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] multiValueMap [RETURN_TYPE] <K,V>   Collection<V>> map [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  Map  map  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new MultiValueMap<K, V> ( collectionClass, new ReflectionFactory<C> ( map )  ) ;^107^^^^^105^108^return new MultiValueMap<K, V> ( map, new ReflectionFactory<C> ( collectionClass )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] multiValueMap [RETURN_TYPE] <K,V,C   Map<K, ? super C> map Class<C> collectionClass [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new MultiValueMap<K, V> ( collectionFactory, map ) ;^124^^^^^122^125^return new MultiValueMap<K, V> ( map, collectionFactory ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] multiValueMap [RETURN_TYPE] <K,V,C   Map<K, ? super C> map Factory<C> collectionFactory [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesView  Object  key  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^final Collection<V> valuesForKey = getCollection ( value ) ;^213^^^^^212^225^final Collection<V> valuesForKey = getCollection ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( valuesForKey != null )  {^214^^^^^212^225^if  ( valuesForKey == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^215^^^^^212^225^return false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^final boolean removed = keysForKey.remove ( value ) ;^217^^^^^212^225^final boolean removed = valuesForKey.remove ( value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Argument_Swapping]^final boolean removed = value.remove ( valuesForKey ) ;^217^^^^^212^225^final boolean removed = valuesForKey.remove ( value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Argument_Swapping]^final boolean removed = valuesForKeysForKey.remove ( value ) ;^217^^^^^212^225^final boolean removed = valuesForKey.remove ( value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( removed != false )  {^218^^^^^212^225^if  ( removed == false )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( removed == true )  {^218^^^^^212^225^if  ( removed == false )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^219^^^^^212^225^return false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^remove ( value ) ;^222^^^^^212^225^remove ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^224^^^^^212^225^return true;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] removeMapping [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  removed  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( pairs == null )  {^239^^^^^237^247^if  ( pairs != null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object value [VARIABLES] Entry  entry  Set  pairs  boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  (  (  ( Collection<V> )  entry.getValue (  )  ) .contains ( key )  )  {^241^^^^^237^247^if  (  (  ( Collection<V> )  entry.getValue (  )  ) .contains ( value )  )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object value [VARIABLES] Entry  entry  Set  pairs  boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  (  (  ( Collection<V> )  value.getValue (  )  ) .contains ( entry )  )  {^241^^^^^237^247^if  (  (  ( Collection<V> )  entry.getValue (  )  ) .contains ( value )  )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object value [VARIABLES] Entry  entry  Set  pairs  boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^242^^^^^237^247^return true;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object value [VARIABLES] Entry  entry  Set  pairs  boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^246^^^^^237^247^return false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object value [VARIABLES] Entry  entry  Set  pairs  boolean  Iterator  iterator  Factory  collectionFactory  Collection  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^boolean result = true;^262^^^^^261^276^boolean result = false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( 3 == null )  {^264^^^^^261^276^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll != null )  {^264^^^^^261^276^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll.size (  )  >= 0 )  {^267^^^^^261^276^if  ( coll.size (  )  > 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( coll.size (  )  > -1 )  {^267^^^^^261^276^if  ( coll.size (  )  > 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^result = false;^270^^^^^261^276^result = true;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Argument_Swapping]^decorated (  ) .put ( coll, key ) ;^269^^^^^261^276^decorated (  ) .put ( key, coll ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^coll = createCollection (  ) ;^265^^^^^261^276^coll = createCollection ( 1 ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^coll = createCollection ( 0 ) ;^265^^^^^261^276^coll = createCollection ( 1 ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^result = null.add (  ( V )  value ) ;^273^^^^^261^276^result = coll.add (  ( V )  value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^return result ? key : null;^275^^^^^261^276^return result ? value : null;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Argument_Swapping]^return value ? result : null;^275^^^^^261^276^return result ? value : null;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] put [RETURN_TYPE] Object   final K key Object value [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( map  &&  MultiMap )  {^292^^^^^291^301^if  ( map instanceof MultiMap )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ?> map [VARIABLES] Entry  entry  boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^put ( this.getKey (  ) , entry.getValue (  )  ) ;^298^^^^^291^301^put ( entry.getKey (  ) , entry.getValue (  )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ?> map [VARIABLES] Entry  entry  boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^putAll ( null.getKey (  ) ,  ( Collection<V> )  entry.getValue (  )  ) ;^294^^^^^291^301^putAll ( entry.getKey (  ) ,  ( Collection<V> )  entry.getValue (  )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ?> map [VARIABLES] Entry  entry  boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Variable_Misuse]^putAll ( 1.getKey (  ) ,  ( Collection<V> )  entry.getValue (  )  ) ;^294^^^^^291^301^putAll ( entry.getKey (  ) ,  ( Collection<V> )  entry.getValue (  )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] void   Map<? extends K, ?> map [VARIABLES] Entry  entry  boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  Object  key  value  Class  clazz  collectionClass  Map  map  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( Collection<Object> )   ( vs == null ? vs :  ( valuesView = new Values (  )  )  ) ;^328^^^^^326^329^return  ( Collection<Object> )   ( vs != null ? vs :  ( valuesView = new Values (  )  )  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] values [RETURN_TYPE] Collection   [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^final Collection<V> coll = getCollection ( value ) ;^339^^^^^338^344^final Collection<V> coll = getCollection ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll != null )  {^340^^^^^338^344^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^341^^^^^338^344^return false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return coll.contains ( key ) ;^343^^^^^338^344^return coll.contains ( value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Argument_Swapping]^return value.contains ( coll ) ;^343^^^^^338^344^return coll.contains ( value ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] containsValue [RETURN_TYPE] boolean   Object key Object value [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return  ( Collection<V> )  decorated (  ) .get ( value ) ;^355^^^^^354^356^return  ( Collection<V> )  decorated (  ) .get ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] getCollection [RETURN_TYPE] Collection   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^final Collection<V> coll = getCollection ( value ) ;^365^^^^^364^370^final Collection<V> coll = getCollection ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] size [RETURN_TYPE] int   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( null == null )  {^366^^^^^364^370^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] size [RETURN_TYPE] int   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll != null )  {^366^^^^^364^370^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] size [RETURN_TYPE] int   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Literal]^return 1;^367^^^^^364^370^return 0;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] size [RETURN_TYPE] int   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( values == null && values.size (  )  == 0 )  {^381^^^^^380^398^if  ( values == null || values.size (  )  == 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( values != null || values.size (  )  == 0 )  {^381^^^^^380^398^if  ( values == null || values.size (  )  == 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( values == null || values.size (  )  != 0 )  {^381^^^^^380^398^if  ( values == null || values.size (  )  == 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^382^^^^^380^398^return false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^boolean result = true;^384^^^^^380^398^boolean result = false;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll != null )  {^386^^^^^380^398^if  ( coll == null )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^result = coll.addAll ( 0 ) ;^395^^^^^380^398^result = coll.addAll ( values ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Argument_Swapping]^result = values.addAll ( coll ) ;^395^^^^^380^398^result = coll.addAll ( values ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( coll.size (  )  >= 0 )  {^389^^^^^380^398^if  ( coll.size (  )  > 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( coll.size (  )  > -1 )  {^389^^^^^380^398^if  ( coll.size (  )  > 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^result = false;^392^^^^^380^398^result = true;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Argument_Swapping]^decorated (  ) .put ( coll, key ) ;^391^^^^^380^398^decorated (  ) .put ( key, coll ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( coll.size (  )  > 1 )  {^389^^^^^380^398^if  ( coll.size (  )  > 0 )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^result = coll.addAll ( null ) ;^395^^^^^380^398^result = coll.addAll ( values ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] putAll [RETURN_TYPE] boolean   final K key Collection<V> values [VARIABLES] boolean  result  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !containsKey ( value )  )  {^407^^^^^406^411^if  ( !containsKey ( key )  )  {^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] iterator [RETURN_TYPE] Iterator   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new ValuesIterator ( value ) ;^410^^^^^406^411^return new ValuesIterator ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] iterator [RETURN_TYPE] Iterator   Object key [VARIABLES] boolean  Iterator  iterator  Factory  collectionFactory  Collection  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return value;^443^^^^^425^454^return input;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] Transformer  transformer  boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  int  count  
[BugLab_Argument_Swapping]^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( transformer ) , key ) ;^451^^^^^425^454^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( key ) , transformer ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] Transformer  transformer  boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  int  count  
[BugLab_Variable_Misuse]^final K key = this.next (  ) ;^435^^^^^425^454^final K key = keyIterator.next (  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] Transformer  transformer  boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  int  count  
[BugLab_Variable_Misuse]^return value;^443^^^^^431^452^return input;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] nextIterator [RETURN_TYPE] Iterator   int count [VARIABLES] Transformer  transformer  boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  int  count  
[BugLab_Argument_Swapping]^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( transformer ) , key ) ;^451^^^^^431^452^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( key ) , transformer ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] nextIterator [RETURN_TYPE] Iterator   int count [VARIABLES] Transformer  transformer  boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  K  key  long  serialVersionUID  int  count  
[BugLab_Variable_Misuse]^return value;^443^^^^^437^449^return input;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] transform [RETURN_TYPE] Entry   final V input [VARIABLES] boolean  V  input  value  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^total += CollectionUtils.size ( value ) ;^464^^^^^461^467^total += CollectionUtils.size ( v ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] totalSize [RETURN_TYPE] int   [VARIABLES] boolean  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  v  value  Class  clazz  collectionClass  long  serialVersionUID  int  total  
[BugLab_Variable_Misuse]^return this.create (  ) ;^480^^^^^479^481^return collectionFactory.create (  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] createCollection [RETURN_TYPE] Collection   final int size [VARIABLES] boolean  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  v  value  Class  clazz  collectionClass  long  serialVersionUID  int  size  
[BugLab_Variable_Misuse]^MultiValueMap.this.remove ( value ) ;^525^^^^^522^527^MultiValueMap.this.remove ( key ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] boolean  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  v  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return null.next (  ) ;^534^^^^^533^535^return iterator.next (  ) ;^[CLASS] MultiValueMap 1 1 1 Values ValuesIterator ReflectionFactory  [METHOD] next [RETURN_TYPE] V   [VARIABLES] boolean  Iterator  iterator  keyIterator  Factory  collectionFactory  Collection  allKeys  coll  values  valuesForKey  valuesView  vs  Object  key  v  value  Class  clazz  collectionClass  long  serialVersionUID  
[BugLab_Variable_Misuse]^return value;^443^^^^^431^452^return input;^[CLASS] 1 1 1  [METHOD] nextIterator [RETURN_TYPE] Iterator   int count [VARIABLES] Transformer  transformer  boolean  V  input  value  K  key  int  count  
[BugLab_Argument_Swapping]^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( transformer ) , key ) ;^451^^^^^431^452^return new TransformIterator<V, Entry<K, V>> ( new ValuesIterator ( key ) , transformer ) ;^[CLASS] 1 1 1  [METHOD] nextIterator [RETURN_TYPE] Iterator   int count [VARIABLES] Transformer  transformer  boolean  V  input  value  K  key  int  count  
[BugLab_Variable_Misuse]^return value;^443^^^^^437^449^return input;^[CLASS] 1 1 1  [METHOD] transform [RETURN_TYPE] Entry   final V input [VARIABLES] boolean  V  input  value  
[BugLab_Variable_Misuse]^return this;^443^^^^^442^444^return input;^[CLASS] 1 1 1  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] boolean  
[BugLab_Variable_Misuse]^return value;^443^^^^^437^449^return input;^[CLASS] 1 1  [METHOD] transform [RETURN_TYPE] Entry   final V input [VARIABLES] boolean  V  input  value  
[BugLab_Variable_Misuse]^return null;^494^^^^^489^495^return chain;^[CLASS] Values  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] boolean  K  k  IteratorChain  chain  
[BugLab_Variable_Misuse]^this.iterator = this.iterator (  ) ;^519^^^^^516^520^this.iterator = values.iterator (  ) ;^[CLASS] ValuesIterator  [METHOD] <init> [RETURN_TYPE] Object)   Object key [VARIABLES] Iterator  iterator  Collection  values  Object  key  boolean  
[BugLab_Variable_Misuse]^return null.newInstance (  ) ;^554^^^^^552^558^return clazz.newInstance (  ) ;^[CLASS] ReflectionFactory  [METHOD] create [RETURN_TYPE] T   [VARIABLES] Class  clazz  boolean  long  serialVersionUID  Exception  ex  
