[P2_Replace_Operator]^if  ( entry != null )  {^42^^^^^41^46^if  ( entry == null )  {^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P8_Replace_Mix]^if  ( entry == this )  {^42^^^^^41^46^if  ( entry == null )  {^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Map Entry must not be null");^42^43^44^^^41^46^if  ( entry == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P16_Remove_Block]^^42^43^44^^^41^46^if  ( entry == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P13_Insert_Block]^if  ( entry == null )  {     throw new IllegalArgumentException ( "Map Entry must not be null" ) ; }^43^^^^^41^46^[Delete]^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P8_Replace_Mix]^this.entry =  null;^45^^^^^41^46^this.entry = entry;^[CLASS] AbstractMapEntryDecorator  [METHOD] <init> [RETURN_TYPE] Map$Entry)   Entry<K, V> entry [VARIABLES] Entry  entry  boolean  
[P7_Replace_Invocation]^return entry.getValue (  ) ;^60^^^^^59^61^return entry.getKey (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] Entry  entry  boolean  
[P14_Delete_Statement]^^60^^^^^59^61^return entry.getKey (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] getKey [RETURN_TYPE] K   [VARIABLES] Entry  entry  boolean  
[P7_Replace_Invocation]^return entry.getKey (  ) ;^64^^^^^63^65^return entry.getValue (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] Entry  entry  boolean  
[P14_Delete_Statement]^^64^^^^^63^65^return entry.getValue (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] Entry  entry  boolean  
[P5_Replace_Variable]^return object.setValue ( entry ) ;^68^^^^^67^69^return entry.setValue ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] setValue [RETURN_TYPE] V   final V object [VARIABLES] Entry  entry  V  object  boolean  
[P7_Replace_Invocation]^return entry.equals ( object ) ;^68^^^^^67^69^return entry.setValue ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] setValue [RETURN_TYPE] V   final V object [VARIABLES] Entry  entry  V  object  boolean  
[P7_Replace_Invocation]^return entry .getValue (  )  ;^68^^^^^67^69^return entry.setValue ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] setValue [RETURN_TYPE] V   final V object [VARIABLES] Entry  entry  V  object  boolean  
[P14_Delete_Statement]^^68^^^^^67^69^return entry.setValue ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] setValue [RETURN_TYPE] V   final V object [VARIABLES] Entry  entry  V  object  boolean  
[P2_Replace_Operator]^if  ( object <= this )  {^73^^^^^72^77^if  ( object == this )  {^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P15_Unwrap_Block]^return true;^73^74^75^^^72^77^if  ( object == this )  { return true; }^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P16_Remove_Block]^^73^74^75^^^72^77^if  ( object == this )  { return true; }^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P3_Replace_Literal]^return false;^74^^^^^72^77^return true;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P5_Replace_Variable]^return object.equals ( entry ) ;^76^^^^^72^77^return entry.equals ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P7_Replace_Invocation]^return entry.setValue ( object ) ;^76^^^^^72^77^return entry.equals ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P7_Replace_Invocation]^return entry .setValue ( null )  ;^76^^^^^72^77^return entry.equals ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P5_Replace_Variable]^return null.equals ( object ) ;^76^^^^^72^77^return entry.equals ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P14_Delete_Statement]^^76^^^^^72^77^return entry.equals ( object ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Entry  entry  Object  object  boolean  
[P7_Replace_Invocation]^return entry.getValue (  ) ;^81^^^^^80^82^return entry.hashCode (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Entry  entry  boolean  
[P14_Delete_Statement]^^81^^^^^80^82^return entry.hashCode (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Entry  entry  boolean  
[P7_Replace_Invocation]^return entry .Object (  )  ;^86^^^^^85^87^return entry.toString (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Entry  entry  boolean  
[P14_Delete_Statement]^^86^^^^^85^87^return entry.toString (  ) ;^[CLASS] AbstractMapEntryDecorator  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Entry  entry  boolean  
