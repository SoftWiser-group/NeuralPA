[P8_Replace_Mix]^private  MapIterator<K, V> iterator;^32^^^^^27^37^private final MapIterator<K, V> iterator;^[CLASS] AbstractMapIteratorDecorator   [VARIABLES] 
[P14_Delete_Statement]^^42^^^^^41^47^super (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P2_Replace_Operator]^if  ( iterator != null )  {^43^^^^^41^47^if  ( iterator == null )  {^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P5_Replace_Variable]^if  ( null == null )  {^43^^^^^41^47^if  ( iterator == null )  {^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P8_Replace_Mix]^if  ( iterator == false )  {^43^^^^^41^47^if  ( iterator == null )  {^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("MapIterator must not be null");^43^44^45^^^41^47^if  ( iterator == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P16_Remove_Block]^^43^44^45^^^41^47^if  ( iterator == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P13_Insert_Block]^if  ( iterator == null )  {     throw new IllegalArgumentException ( "MapIterator must not be null" ) ; }^44^^^^^41^47^[Delete]^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P8_Replace_Mix]^this.iterator =  null;^46^^^^^41^47^this.iterator = iterator;^[CLASS] AbstractMapIteratorDecorator  [METHOD] <init> [RETURN_TYPE] MapIterator)   MapIterator<K, V> iterator [VARIABLES] MapIterator  iterator  boolean  
[P8_Replace_Mix]^return this;^55^^^^^54^56^return iterator;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getMapIterator [RETURN_TYPE] MapIterator   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^return iterator.next (  ) ;^62^^^^^61^63^return iterator.hasNext (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] MapIterator  iterator  boolean  
[P14_Delete_Statement]^^62^^^^^61^63^return iterator.hasNext (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] hasNext [RETURN_TYPE] boolean   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^return iterator.hasNext (  ) ;^67^^^^^66^68^return iterator.next (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] next [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P8_Replace_Mix]^return null.next (  ) ;^67^^^^^66^68^return iterator.next (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] next [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P14_Delete_Statement]^^67^^^^^66^68^return iterator.next (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] next [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^iterator.next (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^iterator .getValue (  )  ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P14_Delete_Statement]^^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P11_Insert_Donor_Statement]^return iterator.next (  ) ;iterator.remove (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P11_Insert_Donor_Statement]^return iterator.getKey (  ) ;iterator.remove (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P11_Insert_Donor_Statement]^return iterator.getValue (  ) ;iterator.remove (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P11_Insert_Donor_Statement]^return iterator.setValue ( obj ) ;iterator.remove (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P11_Insert_Donor_Statement]^return iterator.hasNext (  ) ;iterator.remove (  ) ;^72^^^^^71^73^iterator.remove (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] remove [RETURN_TYPE] void   [VARIABLES] MapIterator  iterator  boolean  
[P5_Replace_Variable]^return this.getKey (  ) ;^77^^^^^76^78^return iterator.getKey (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^return iterator.getValue (  ) ;^77^^^^^76^78^return iterator.getKey (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P8_Replace_Mix]^return 4.getKey (  ) ;^77^^^^^76^78^return iterator.getKey (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P14_Delete_Statement]^^77^^^^^76^78^return iterator.getKey (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] MapIterator  iterator  boolean  
[P7_Replace_Invocation]^return iterator.getKey (  ) ;^82^^^^^81^83^return iterator.getValue (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] MapIterator  iterator  boolean  
[P14_Delete_Statement]^^82^^^^^81^83^return iterator.getValue (  ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] MapIterator  iterator  boolean  
[P5_Replace_Variable]^return obj.setValue ( iterator ) ;^87^^^^^86^88^return iterator.setValue ( obj ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] setValue [RETURN_TYPE] V   final V obj [VARIABLES] MapIterator  iterator  V  obj  boolean  
[P7_Replace_Invocation]^return iterator .getValue (  )  ;^87^^^^^86^88^return iterator.setValue ( obj ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] setValue [RETURN_TYPE] V   final V obj [VARIABLES] MapIterator  iterator  V  obj  boolean  
[P14_Delete_Statement]^^87^^^^^86^88^return iterator.setValue ( obj ) ;^[CLASS] AbstractMapIteratorDecorator  [METHOD] setValue [RETURN_TYPE] V   final V obj [VARIABLES] MapIterator  iterator  V  obj  boolean  