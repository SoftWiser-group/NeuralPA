[P1_Replace_Type]^private static final  int  serialVersionUID = -8453869361373831205L;^36^^^^^31^41^private static final long serialVersionUID = -8453869361373831205L;^[CLASS] TiedMapEntry   [VARIABLES] 
[P8_Replace_Mix]^private static  long serialVersionUID = -8453869361373831205;^36^^^^^31^41^private static final long serialVersionUID = -8453869361373831205L;^[CLASS] TiedMapEntry   [VARIABLES] 
[P8_Replace_Mix]^private  Map<K, V> map;^39^^^^^34^44^private final Map<K, V> map;^[CLASS] TiedMapEntry   [VARIABLES] 
[P14_Delete_Statement]^^51^52^^^^50^54^super (  ) ; this.map = map;^[CLASS] TiedMapEntry  [METHOD] <init> [RETURN_TYPE] Map,K)   Map<K, V> map final K key [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P8_Replace_Mix]^this.map =  null;^52^^^^^50^54^this.map = map;^[CLASS] TiedMapEntry  [METHOD] <init> [RETURN_TYPE] Map,K)   Map<K, V> map final K key [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P11_Insert_Donor_Statement]^this.key = key;this.map = map;^52^^^^^50^54^this.map = map;^[CLASS] TiedMapEntry  [METHOD] <init> [RETURN_TYPE] Map,K)   Map<K, V> map final K key [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P8_Replace_Mix]^this.key =  null;^53^^^^^50^54^this.key = key;^[CLASS] TiedMapEntry  [METHOD] <init> [RETURN_TYPE] Map,K)   Map<K, V> map final K key [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P11_Insert_Donor_Statement]^this.map = map;this.key = key;^53^^^^^50^54^this.key = key;^[CLASS] TiedMapEntry  [METHOD] <init> [RETURN_TYPE] Map,K)   Map<K, V> map final K key [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^return key.get ( map ) ;^73^^^^^72^74^return map.get ( key ) ;^[CLASS] TiedMapEntry  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^73^^^^^72^74^return map.get ( key ) ;^[CLASS] TiedMapEntry  [METHOD] getValue [RETURN_TYPE] V   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^if  ( value >= this )  {^84^^^^^83^88^if  ( value == this )  {^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P9_Replace_Statement]^if  ( obj == this )  {^84^^^^^83^88^if  ( value == this )  {^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Cannot set value to this map entry");^84^85^86^^^83^88^if  ( value == this )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P16_Remove_Block]^^84^85^86^^^83^88^if  ( value == this )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P13_Insert_Block]^if  ( value ==  ( this )  )  {     throw new IllegalArgumentException ( "Cannot set value to this map entry" ) ; }^85^^^^^83^88^[Delete]^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P8_Replace_Mix]^return 0;^85^^^^^83^88^throw new IllegalArgumentException  (" ")  ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put ( key ) ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put (  value ) ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return map.put ( value, key ) ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return value.put ( key, map ) ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P5_Replace_Variable]^return key.put ( map, value ) ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P8_Replace_Mix]^return map .get ( 2 )  ;^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P14_Delete_Statement]^^87^^^^^83^88^return map.put ( key, value ) ;^[CLASS] TiedMapEntry  [METHOD] setValue [RETURN_TYPE] V   final V value [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  V  value  
[P2_Replace_Operator]^if  ( obj <= this )  {^100^^^^^99^111^if  ( obj == this )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^if  ( value == this )  {^100^^^^^99^111^if  ( obj == this )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P15_Unwrap_Block]^return true;^100^101^102^^^99^111^if  ( obj == this )  { return true; }^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P16_Remove_Block]^^100^101^102^^^99^111^if  ( obj == this )  { return true; }^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^return false;^101^^^^^99^111^return true;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^if  ( obj instanceof Map.Entry != false )  {^103^^^^^99^111^if  ( obj instanceof Map.Entry == false )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^if  ( obj  >  Map.Entry == false )  {^103^^^^^99^111^if  ( obj instanceof Map.Entry == false )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^if  ( obj instanceof Map.Entry == true )  {^103^^^^^99^111^if  ( obj instanceof Map.Entry == false )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^if  ( value instanceof Map.Entry == false )  {^103^^^^^99^111^if  ( obj instanceof Map.Entry == false )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^if  ( obj instanceof Entry )  {^103^^^^^99^111^if  ( obj instanceof Map.Entry == false )  {^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P15_Unwrap_Block]^return false;^103^104^105^^^99^111^if  ( obj instanceof Map.Entry == false )  { return false; }^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P16_Remove_Block]^^103^104^105^^^99^111^if  ( obj instanceof Map.Entry == false )  { return false; }^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^return true;^104^^^^^99^111^return false;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^final Object value = getKey (  ) ;^107^^^^^99^111^final Object value = getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^107^^^^^99^111^final Object value = getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  || ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return ( key != null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value != null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^return ( other == null ? key.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^return ( key == null ? value.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( other == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^return ( value == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( key == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^return ( key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return ( key == null ? other .getValue (  )   == null : key.equals ( other^108^109^110^^^99^111^return ( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^( key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null^109^110^^^^99^111^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P8_Replace_Mix]^( key == null ? other .getValue (  )   == null : key.equals ( other^109^110^^^^99^111^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^109^110^^^^99^111^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P11_Insert_Donor_Statement]^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^109^110^^^^99^111^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^( other == null ? key.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^109^110^^^^99^111^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^( value.equals ( other.getValue (  )  )  ) ;^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P11_Insert_Donor_Statement]^( key == null ? other.getKey (  )  == null : key.equals ( other.getKey (  )  )  )  && ( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^( obj == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P5_Replace_Variable]^( other == null ? value.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P8_Replace_Mix]^( value == null ? 0.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^110^^^^^99^111^( value == null ? other.getValue (  )  == null : value.equals ( other.getValue (  )  )  ) ;^[CLASS] TiedMapEntry  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  value  Entry  other  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^final Object value = getKey (  ) ;^122^^^^^121^125^final Object value = getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^122^^^^^121^125^final Object value = getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return  ( getKey (  )  != null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value != null ? 0 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^return  ( getKey (  )  == null ? 5 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 5 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^return  ( getKey (  )  == null ?  : getKey (  ) .hashCode (  )  )  ^ ( value == null ?  : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^return  ( getKey ( getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return  ( getValue (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return  ( getKey (  )  == null ? 0 : getKey (  ) .equals (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return  ( getKey (  )  == null ? 0 : getKey (  )  .Object (  )   )  ^ ( value == null ? 0 : value^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P8_Replace_Mix]^return  ( getValue (  )  == false ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == false ? 0 : value.hashCode (  )  ) ;^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^123^124^^^^121^125^return  ( getKey (  )  == null ? 0 : getKey (  ) .hashCode (  )  )  ^ ( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P6_Replace_Expression]^( value.hashCode (  )  ) ;^124^^^^^121^125^( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^( value == null ? 0 : value.equals (  )  ) ;^124^^^^^121^125^( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^( value == null ? 0 : value .Object (  )   ) ;^124^^^^^121^125^( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^124^^^^^121^125^( value == null ? 0 : value.hashCode (  )  ) ;^[CLASS] TiedMapEntry  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Object  value  K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return getKey (  ||  )  + "=" + getValue (  ) ;^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P2_Replace_Operator]^return getKey (  )   ==  "=" + getValue (  ) ;^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P3_Replace_Literal]^return getKey (  )  + "" + getValue (  ) ;^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return getValue (  )  + "=" + getValue (  ) ;^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P7_Replace_Invocation]^return getKey (  )  + "=" + getKey (  ) ;^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
[P14_Delete_Statement]^^134^^^^^133^135^return getKey (  )  + "=" + getValue (  ) ;^[CLASS] TiedMapEntry  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] K  key  boolean  Map  map  long  serialVersionUID  
