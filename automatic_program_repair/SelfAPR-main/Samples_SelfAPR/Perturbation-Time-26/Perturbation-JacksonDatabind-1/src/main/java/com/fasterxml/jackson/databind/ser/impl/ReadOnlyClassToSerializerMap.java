[P8_Replace_Mix]^protected  JsonSerializerMap _map;^20^^^^^15^25^protected final JsonSerializerMap _map;^[CLASS] ReadOnlyClassToSerializerMap   [VARIABLES] 
[P8_Replace_Mix]^protected TypeKey _cacheKey = false;^27^^^^^22^32^protected TypeKey _cacheKey = null;^[CLASS] ReadOnlyClassToSerializerMap   [VARIABLES] 
[P5_Replace_Variable]^_map = _map;^31^^^^^29^32^_map = map;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] <init> [RETURN_TYPE] JsonSerializerMap)   JsonSerializerMap map [VARIABLES] JsonSerializerMap  _map  map  TypeKey  _cacheKey  boolean  
[P8_Replace_Mix]^_map =  null;^31^^^^^29^32^_map = map;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] <init> [RETURN_TYPE] JsonSerializerMap)   JsonSerializerMap map [VARIABLES] JsonSerializerMap  _map  map  TypeKey  _cacheKey  boolean  
[P4_Replace_Constructor]^return return  new JsonSerializerMap ( src )  ;^36^^^^^34^37^return new ReadOnlyClassToSerializerMap ( _map ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] instance [RETURN_TYPE] ReadOnlyClassToSerializerMap   [VARIABLES] JsonSerializerMap  _map  map  TypeKey  _cacheKey  boolean  
[P5_Replace_Variable]^return new ReadOnlyClassToSerializerMap ( map ) ;^36^^^^^34^37^return new ReadOnlyClassToSerializerMap ( _map ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] instance [RETURN_TYPE] ReadOnlyClassToSerializerMap   [VARIABLES] JsonSerializerMap  _map  map  TypeKey  _cacheKey  boolean  
[P8_Replace_Mix]^return  new JsonSerializerMap ( src )  ;^36^^^^^34^37^return new ReadOnlyClassToSerializerMap ( _map ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] instance [RETURN_TYPE] ReadOnlyClassToSerializerMap   [VARIABLES] JsonSerializerMap  _map  map  TypeKey  _cacheKey  boolean  
[P8_Replace_Mix]^return new ReadOnlyClassToSerializerMap ( new JsonSerializerMap ( 0 )  ) ;^46^^^^^44^47^return new ReadOnlyClassToSerializerMap ( new JsonSerializerMap ( src )  ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] from [RETURN_TYPE] ReadOnlyClassToSerializerMap   Object>> src [VARIABLES] TypeKey  _cacheKey  boolean  JsonSerializerMap  _map  map  HashMap  src  
[P5_Replace_Variable]^return new ReadOnlyClassToSerializerMap ( new JsonSerializerMap ( 3 )  ) ;^46^^^^^44^47^return new ReadOnlyClassToSerializerMap ( new JsonSerializerMap ( src )  ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] from [RETURN_TYPE] ReadOnlyClassToSerializerMap   Object>> src [VARIABLES] TypeKey  _cacheKey  boolean  JsonSerializerMap  _map  map  HashMap  src  
[P2_Replace_Operator]^if  ( _cacheKey != null )  {^51^^^^^49^57^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^if  ( _cacheKey == this )  {^51^^^^^49^57^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P15_Unwrap_Block]^_cacheKey = new com.fasterxml.jackson.databind.ser.SerializerCache.TypeKey(type, true);^51^52^53^54^55^49^57^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( type, true ) ; } else { _cacheKey.resetTyped ( type ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P16_Remove_Block]^^51^52^53^54^55^49^57^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( type, true ) ; } else { _cacheKey.resetTyped ( type ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^51^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^51^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^51^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P7_Replace_Invocation]^_cacheKey.resetUntyped ( type ) ;^54^^^^^49^57^_cacheKey.resetTyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^54^^^^^49^57^_cacheKey.resetTyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( cls ) ;_cacheKey.resetTyped ( type ) ;^54^^^^^49^57^_cacheKey.resetTyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( type ) ;_cacheKey.resetTyped ( type ) ;^54^^^^^49^57^_cacheKey.resetTyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( cls ) ;_cacheKey.resetTyped ( type ) ;^54^^^^^49^57^_cacheKey.resetTyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P3_Replace_Literal]^_cacheKey = new TypeKey ( type, false ) ;^52^^^^^49^57^_cacheKey = new TypeKey ( type, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, true ) ;_cacheKey = new TypeKey ( type, true ) ;^52^^^^^49^57^_cacheKey = new TypeKey ( type, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, false ) ;_cacheKey = new TypeKey ( type, true ) ;^52^^^^^49^57^_cacheKey = new TypeKey ( type, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, false ) ;_cacheKey = new TypeKey ( type, true ) ;^52^^^^^49^57^_cacheKey = new TypeKey ( type, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^52^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^52^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^52^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^52^^^^^49^57^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return map.find ( _cacheKey ) ;^56^^^^^49^57^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return _cacheKey.find ( _map ) ;^56^^^^^49^57^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^56^^^^^49^57^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P2_Replace_Operator]^if  ( _cacheKey != null )  {^61^^^^^59^67^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^if  ( _cacheKey == this )  {^61^^^^^59^67^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P15_Unwrap_Block]^_cacheKey = new com.fasterxml.jackson.databind.ser.SerializerCache.TypeKey(cls, true);^61^62^63^64^65^59^67^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( cls, true ) ; } else { _cacheKey.resetTyped ( cls ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P16_Remove_Block]^^61^62^63^64^65^59^67^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( cls, true ) ; } else { _cacheKey.resetTyped ( cls ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^61^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^61^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^61^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P7_Replace_Invocation]^_cacheKey.resetUntyped ( cls ) ;^64^^^^^59^67^_cacheKey.resetTyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^64^^^^^59^67^_cacheKey.resetTyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( type ) ;_cacheKey.resetTyped ( cls ) ;^64^^^^^59^67^_cacheKey.resetTyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( type ) ;_cacheKey.resetTyped ( cls ) ;^64^^^^^59^67^_cacheKey.resetTyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( cls ) ;_cacheKey.resetTyped ( cls ) ;^64^^^^^59^67^_cacheKey.resetTyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P3_Replace_Literal]^_cacheKey = new TypeKey ( cls, false ) ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P4_Replace_Constructor]^_cacheKey = _cacheKey =  new TypeKey ( type, false )  ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^_cacheKey =  new TypeKey ( type, false )  ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, true ) ;_cacheKey = new TypeKey ( cls, true ) ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, false ) ;_cacheKey = new TypeKey ( cls, true ) ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, false ) ;_cacheKey = new TypeKey ( cls, true ) ;^62^^^^^59^67^_cacheKey = new TypeKey ( cls, true ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^62^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^62^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^62^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^62^^^^^59^67^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return map.find ( _cacheKey ) ;^66^^^^^59^67^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return _cacheKey.find ( _map ) ;^66^^^^^59^67^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^66^^^^^59^67^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] typedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P2_Replace_Operator]^if  ( _cacheKey != null )  {^71^^^^^69^77^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^if  ( _cacheKey == true )  {^71^^^^^69^77^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P15_Unwrap_Block]^_cacheKey = new com.fasterxml.jackson.databind.ser.SerializerCache.TypeKey(type, false);^71^72^73^74^75^69^77^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( type, false ) ; } else { _cacheKey.resetUntyped ( type ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P16_Remove_Block]^^71^72^73^74^75^69^77^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( type, false ) ; } else { _cacheKey.resetUntyped ( type ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^71^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^71^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^71^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P7_Replace_Invocation]^_cacheKey.resetTyped ( type ) ;^74^^^^^69^77^_cacheKey.resetUntyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^74^^^^^69^77^_cacheKey.resetUntyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( type ) ;_cacheKey.resetUntyped ( type ) ;^74^^^^^69^77^_cacheKey.resetUntyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( cls ) ;_cacheKey.resetUntyped ( type ) ;^74^^^^^69^77^_cacheKey.resetUntyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( cls ) ;_cacheKey.resetUntyped ( type ) ;^74^^^^^69^77^_cacheKey.resetUntyped ( type ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P3_Replace_Literal]^_cacheKey = new TypeKey ( type, true ) ;^72^^^^^69^77^_cacheKey = new TypeKey ( type, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, true ) ;_cacheKey = new TypeKey ( type, false ) ;^72^^^^^69^77^_cacheKey = new TypeKey ( type, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, true ) ;_cacheKey = new TypeKey ( type, false ) ;^72^^^^^69^77^_cacheKey = new TypeKey ( type, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, false ) ;_cacheKey = new TypeKey ( type, false ) ;^72^^^^^69^77^_cacheKey = new TypeKey ( type, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^72^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^72^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^72^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^72^^^^^69^77^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return map.find ( _cacheKey ) ;^76^^^^^69^77^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return _cacheKey.find ( _map ) ;^76^^^^^69^77^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^76^^^^^69^77^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   JavaType type [VARIABLES] TypeKey  _cacheKey  JavaType  type  boolean  JsonSerializerMap  _map  map  
[P2_Replace_Operator]^if  ( _cacheKey != null )  {^81^^^^^79^87^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^if  ( _cacheKey == false )  {^81^^^^^79^87^if  ( _cacheKey == null )  {^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P15_Unwrap_Block]^_cacheKey = new com.fasterxml.jackson.databind.ser.SerializerCache.TypeKey(cls, false);^81^82^83^84^85^79^87^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( cls, false ) ; } else { _cacheKey.resetUntyped ( cls ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P16_Remove_Block]^^81^82^83^84^85^79^87^if  ( _cacheKey == null )  { _cacheKey = new TypeKey ( cls, false ) ; } else { _cacheKey.resetUntyped ( cls ) ; }^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^81^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^81^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^81^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P7_Replace_Invocation]^_cacheKey.resetTyped ( cls ) ;^84^^^^^79^87^_cacheKey.resetUntyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^84^^^^^79^87^_cacheKey.resetUntyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( type ) ;_cacheKey.resetUntyped ( cls ) ;^84^^^^^79^87^_cacheKey.resetUntyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetTyped ( cls ) ;_cacheKey.resetUntyped ( cls ) ;^84^^^^^79^87^_cacheKey.resetUntyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey.resetUntyped ( type ) ;_cacheKey.resetUntyped ( cls ) ;^84^^^^^79^87^_cacheKey.resetUntyped ( cls ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P3_Replace_Literal]^_cacheKey = new TypeKey ( cls, true ) ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P4_Replace_Constructor]^_cacheKey = _cacheKey =  new TypeKey ( type, false )  ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^_cacheKey = new TypeKey ( null, false ) ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P8_Replace_Mix]^_cacheKey =  new TypeKey ( type, false )  ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( cls, true ) ;_cacheKey = new TypeKey ( cls, false ) ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, true ) ;_cacheKey = new TypeKey ( cls, false ) ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P11_Insert_Donor_Statement]^_cacheKey = new TypeKey ( type, false ) ;_cacheKey = new TypeKey ( cls, false ) ;^82^^^^^79^87^_cacheKey = new TypeKey ( cls, false ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, false ) ; }else {     _cacheKey.resetUntyped ( cls ) ; }^82^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( cls, true ) ; }else {     _cacheKey.resetTyped ( cls ) ; }^82^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, true ) ; }else {     _cacheKey.resetTyped ( type ) ; }^82^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P13_Insert_Block]^if  (  ( _cacheKey )  == null )  {     _cacheKey = new TypeKey ( type, false ) ; }else {     _cacheKey.resetUntyped ( type ) ; }^82^^^^^79^87^[Delete]^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return map.find ( _cacheKey ) ;^86^^^^^79^87^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P5_Replace_Variable]^return _cacheKey.find ( _map ) ;^86^^^^^79^87^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
[P14_Delete_Statement]^^86^^^^^79^87^return _map.find ( _cacheKey ) ;^[CLASS] ReadOnlyClassToSerializerMap  [METHOD] untypedValueSerializer [RETURN_TYPE] JsonSerializer   Class<?> cls [VARIABLES] TypeKey  _cacheKey  Class  cls  boolean  JsonSerializerMap  _map  map  
