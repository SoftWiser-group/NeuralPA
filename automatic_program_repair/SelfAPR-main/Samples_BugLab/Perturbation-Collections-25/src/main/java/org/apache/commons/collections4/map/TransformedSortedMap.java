[BugLab_Argument_Swapping]^super ( keyTransformer, map, valueTransformer ) ;^118^^^^^115^119^super ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] <init> [RETURN_TYPE] Transformer)   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^super ( map, valueTransformer, keyTransformer ) ;^118^^^^^115^119^super ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] <init> [RETURN_TYPE] Transformer)   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( keyTransformer, map, valueTransformer ) ;^69^^^^^66^70^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformingSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^69^^^^^66^70^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformingSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Variable_Misuse]^return new TransformedSortedMap<K, V> ( map, this, valueTransformer ) ;^69^^^^^66^70^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformingSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^69^^^^^66^70^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformingSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^new TransformedSortedMap<K, V> ( keyTransformer, map, valueTransformer ) ;^94^^^^^89^101^new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Argument_Swapping]^new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^94^^^^^89^101^new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Argument_Swapping]^new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^94^^^^^89^101^new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Argument_Swapping]^final TransformedSortedMap<K, V> decorated = new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^93^94^^^^89^101^final TransformedSortedMap<K, V> decorated = new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Argument_Swapping]^final TransformedSortedMap<K, V> decorated = new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^93^94^^^^89^101^final TransformedSortedMap<K, V> decorated = new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Wrong_Operator]^if  ( map.size (  )  < 0 )  {^95^^^^^89^101^if  ( map.size (  )  > 0 )  {^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Wrong_Literal]^if  ( map.size (  )  > -1 )  {^95^^^^^89^101^if  ( map.size (  )  > 0 )  {^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Argument_Swapping]^final Map<K, V> transformed = map.transformMap ( decorated ) ;^96^^^^^89^101^final Map<K, V> transformed = decorated.transformMap ( map ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Variable_Misuse]^decorated.3 (  ) .putAll ( transformed ) ;^98^^^^^89^101^decorated.decorated (  ) .putAll ( transformed ) ;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Variable_Misuse]^return 0;^100^^^^^89^101^return decorated;^[CLASS] TransformedSortedMap  [METHOD] transformedSortedMap [RETURN_TYPE] <K,V>   SortedMap<K, V> map Transformer<? super K, ? extends K> keyTransformer Transformer<? super V, ? extends V> valueTransformer [VARIABLES] Transformer  keyTransformer  valueTransformer  boolean  Map  transformed  long  serialVersionUID  SortedMap  map  TransformedSortedMap  decorated  
[BugLab_Variable_Misuse]^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( toKey, toKey ) ;^145^^^^^144^147^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( fromKey, toKey ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Variable_Misuse]^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( fromKey, fromKey ) ;^145^^^^^144^147^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( fromKey, toKey ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( toKey, fromKey ) ;^145^^^^^144^147^final SortedMap<K, V> map = getSortedMap (  ) .subMap ( fromKey, toKey ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Variable_Misuse]^return new TransformedSortedMap<K, V> ( null, keyTransformer, valueTransformer ) ;^146^^^^^144^147^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^146^^^^^144^147^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^146^^^^^144^147^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] subMap [RETURN_TYPE] SortedMap   final K fromKey final K toKey [VARIABLES] K  fromKey  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( keyTransformer, map, valueTransformer ) ;^151^^^^^149^152^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] K  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^151^^^^^149^152^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] K  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^151^^^^^149^152^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] headMap [RETURN_TYPE] SortedMap   final K toKey [VARIABLES] K  toKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( keyTransformer, map, valueTransformer ) ;^156^^^^^154^157^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] K  fromKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( map, valueTransformer, keyTransformer ) ;^156^^^^^154^157^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] K  fromKey  boolean  long  serialVersionUID  SortedMap  map  
[BugLab_Argument_Swapping]^return new TransformedSortedMap<K, V> ( valueTransformer, keyTransformer, map ) ;^156^^^^^154^157^return new TransformedSortedMap<K, V> ( map, keyTransformer, valueTransformer ) ;^[CLASS] TransformedSortedMap  [METHOD] tailMap [RETURN_TYPE] SortedMap   final K fromKey [VARIABLES] K  fromKey  boolean  long  serialVersionUID  SortedMap  map  
