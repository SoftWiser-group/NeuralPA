[P3_Replace_Literal]^public final static BooleanNode TRUE = new BooleanNode ( false ) ;^19^^^^^14^24^public final static BooleanNode TRUE = new BooleanNode ( true ) ;^[CLASS] BooleanNode   [VARIABLES] 
[P3_Replace_Literal]^public final static BooleanNode FALSE = new BooleanNode ( true ) ;^20^^^^^15^25^public final static BooleanNode FALSE = new BooleanNode ( false ) ;^[CLASS] BooleanNode   [VARIABLES] 
[P8_Replace_Mix]^private  boolean _value;^22^^^^^17^27^private final boolean _value;^[CLASS] BooleanNode   [VARIABLES] 
[P5_Replace_Variable]^private BooleanNode ( boolean _value )  { _value = v; }^24^^^^^19^29^private BooleanNode ( boolean v )  { _value = v; }^[CLASS] BooleanNode  [METHOD] <init> [RETURN_TYPE] BooleanNode(boolean)   boolean v [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  v  
[P8_Replace_Mix]^private BooleanNode ( boolean v )  { _value =  null; }^24^^^^^19^29^private BooleanNode ( boolean v )  { _value = v; }^[CLASS] BooleanNode  [METHOD] <init> [RETURN_TYPE] BooleanNode(boolean)   boolean v [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  v  
[P5_Replace_Variable]^public static BooleanNode getTrue (  )  { return FALSE; }^26^^^^^21^31^public static BooleanNode getTrue (  )  { return TRUE; }^[CLASS] BooleanNode  [METHOD] getTrue [RETURN_TYPE] BooleanNode   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  v  
[P5_Replace_Variable]^public static BooleanNode getFalse (  )  { return TRUE; }^27^^^^^22^32^public static BooleanNode getFalse (  )  { return FALSE; }^[CLASS] BooleanNode  [METHOD] getFalse [RETURN_TYPE] BooleanNode   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  v  
[P5_Replace_Variable]^public static BooleanNode valueOf ( voolean b )  { return b ? TRUE : FALSE; }^29^^^^^24^34^public static BooleanNode valueOf ( boolean b )  { return b ? TRUE : FALSE; }^[CLASS] BooleanNode  [METHOD] valueOf [RETURN_TYPE] BooleanNode   boolean b [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^public static BooleanNode valueOf ( TRUEoolean b )  { return b ? b : FALSE; }^29^^^^^24^34^public static BooleanNode valueOf ( boolean b )  { return b ? TRUE : FALSE; }^[CLASS] BooleanNode  [METHOD] valueOf [RETURN_TYPE] BooleanNode   boolean b [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^public static BooleanNode valueOf ( boolean b )  { return b ? FALSE : TRUE; }^29^^^^^24^34^public static BooleanNode valueOf ( boolean b )  { return b ? TRUE : FALSE; }^[CLASS] BooleanNode  [METHOD] valueOf [RETURN_TYPE] BooleanNode   boolean b [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^public static BooleanNode valueOf ( FALSEoolean b )  { return b ? TRUE : b; }^29^^^^^24^34^public static BooleanNode valueOf ( boolean b )  { return b ? TRUE : FALSE; }^[CLASS] BooleanNode  [METHOD] valueOf [RETURN_TYPE] BooleanNode   boolean b [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P8_Replace_Mix]^public static BooleanNode valueOf ( boolean b )  { return b ? FALSE : FALSE; }^29^^^^^24^34^public static BooleanNode valueOf ( boolean b )  { return b ? TRUE : FALSE; }^[CLASS] BooleanNode  [METHOD] valueOf [RETURN_TYPE] BooleanNode   boolean b [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return TRUE;^33^^^^^32^34^return JsonNodeType.BOOLEAN;^[CLASS] BooleanNode  [METHOD] getNodeType [RETURN_TYPE] JsonNodeType   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return v ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^37^^^^^36^38^return _value ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^[CLASS] BooleanNode  [METHOD] asToken [RETURN_TYPE] JsonToken   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return _value ? TRUE : JsonToken.VALUE_FALSE;^37^^^^^36^38^return _value ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^[CLASS] BooleanNode  [METHOD] asToken [RETURN_TYPE] JsonToken   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return _value ? JsonToken.VALUE_TRUE : TRUE;^37^^^^^36^38^return _value ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^[CLASS] BooleanNode  [METHOD] asToken [RETURN_TYPE] JsonToken   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return JsonToken.VALUE_FALSE ? JsonToken.VALUE_TRUE : _value;^37^^^^^36^38^return _value ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^[CLASS] BooleanNode  [METHOD] asToken [RETURN_TYPE] JsonToken   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return _value ? JsonToken.VALUE_FALSE : JsonToken.VALUE_TRUE;^37^^^^^36^38^return _value ? JsonToken.VALUE_TRUE : JsonToken.VALUE_FALSE;^[CLASS] BooleanNode  [METHOD] asToken [RETURN_TYPE] JsonToken   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return v;^42^^^^^41^43^return _value;^[CLASS] BooleanNode  [METHOD] booleanValue [RETURN_TYPE] boolean   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P3_Replace_Literal]^return _value ? "" : "false";^47^^^^^46^48^return _value ? "true" : "false";^[CLASS] BooleanNode  [METHOD] asText [RETURN_TYPE] String   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P3_Replace_Literal]^return _value ? "true" : "fa";^47^^^^^46^48^return _value ? "true" : "false";^[CLASS] BooleanNode  [METHOD] asText [RETURN_TYPE] String   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return v ? "true" : "false";^47^^^^^46^48^return _value ? "true" : "false";^[CLASS] BooleanNode  [METHOD] asText [RETURN_TYPE] String   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return v;^52^^^^^51^53^return _value;^[CLASS] BooleanNode  [METHOD] asBoolean [RETURN_TYPE] boolean   [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  v  
[P5_Replace_Variable]^return v;^57^^^^^56^58^return _value;^[CLASS] BooleanNode  [METHOD] asBoolean [RETURN_TYPE] boolean   boolean defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^return _value ? defaultValue : 0;^62^^^^^61^63^return _value ? 1 : 0;^[CLASS] BooleanNode  [METHOD] asInt [RETURN_TYPE] int   int defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  int  defaultValue  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^return _value ? 1 : defaultValue;^62^^^^^61^63^return _value ? 1 : 0;^[CLASS] BooleanNode  [METHOD] asInt [RETURN_TYPE] int   int defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  int  defaultValue  boolean  _value  b  defaultValue  v  
[P5_Replace_Variable]^return v ? 1 : 0;^62^^^^^61^63^return _value ? 1 : 0;^[CLASS] BooleanNode  [METHOD] asInt [RETURN_TYPE] int   int defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  int  defaultValue  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^return _value ? 1L : 0;^62^^^^^61^63^return _value ? 1 : 0;^[CLASS] BooleanNode  [METHOD] asInt [RETURN_TYPE] int   int defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  int  defaultValue  boolean  _value  b  defaultValue  v  
[P5_Replace_Variable]^return v ? 1L : 0L;^66^^^^^65^67^return _value ? 1L : 0L;^[CLASS] BooleanNode  [METHOD] asLong [RETURN_TYPE] long   long defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  long  defaultValue  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^return _value ? 1 : 0L;^66^^^^^65^67^return _value ? 1L : 0L;^[CLASS] BooleanNode  [METHOD] asLong [RETURN_TYPE] long   long defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  long  defaultValue  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^return _value ? 2.0 : 0.0;^70^^^^^69^71^return _value ? 1.0 : 0.0;^[CLASS] BooleanNode  [METHOD] asDouble [RETURN_TYPE] double   double defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  double  defaultValue  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^return _value ? 1.0 : NaN;^70^^^^^69^71^return _value ? 1.0 : 0.0;^[CLASS] BooleanNode  [METHOD] asDouble [RETURN_TYPE] double   double defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  double  defaultValue  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^return _value ? 3.0d : 0.0;^70^^^^^69^71^return _value ? 1.0 : 0.0;^[CLASS] BooleanNode  [METHOD] asDouble [RETURN_TYPE] double   double defaultValue [VARIABLES] BooleanNode  FALSE  TRUE  double  defaultValue  boolean  _value  b  defaultValue  v  
[P5_Replace_Variable]^jg.writeBoolean ( v ) ;^77^^^^^74^78^jg.writeBoolean ( _value ) ;^[CLASS] BooleanNode  [METHOD] serialize [RETURN_TYPE] void   JsonGenerator jg SerializerProvider provider [VARIABLES] JsonGenerator  jg  boolean  _value  b  defaultValue  v  SerializerProvider  provider  BooleanNode  FALSE  TRUE  
[P14_Delete_Statement]^^77^^^^^74^78^jg.writeBoolean ( _value ) ;^[CLASS] BooleanNode  [METHOD] serialize [RETURN_TYPE] void   JsonGenerator jg SerializerProvider provider [VARIABLES] JsonGenerator  jg  boolean  _value  b  defaultValue  v  SerializerProvider  provider  BooleanNode  FALSE  TRUE  
[P2_Replace_Operator]^if  ( o <= this )  return true;^87^^^^^81^93^if  ( o == this )  return true;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^if  ( o == this )  return false;^87^^^^^81^93^if  ( o == this )  return true;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P15_Unwrap_Block]^return true;^87^88^89^90^91^81^93^if  ( o == this )  return true; if  ( o == null )  return false; if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P16_Remove_Block]^^87^88^89^90^91^81^93^if  ( o == this )  return true; if  ( o == null )  return false; if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P2_Replace_Operator]^if  ( o != null )  return false;^88^^^^^81^93^if  ( o == null )  return false;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^if  ( o == null )  return true;^88^^^^^81^93^if  ( o == null )  return false;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^if  ( o == true )  return false;^88^^^^^81^93^if  ( o == null )  return false;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P15_Unwrap_Block]^return false;^88^89^90^91^^81^93^if  ( o == null )  return false; if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P16_Remove_Block]^^88^89^90^91^^81^93^if  ( o == null )  return false; if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P2_Replace_Operator]^if  ( o.getClass (  )  >= getClass (  )  )  {^89^^^^^81^93^if  ( o.getClass (  )  != getClass (  )  )  {^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^if  ( o.getClass (  )  = getClass (  )  )  {^89^^^^^81^93^if  ( o.getClass (  )  != getClass (  )  )  {^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P15_Unwrap_Block]^return false;^89^90^91^^^81^93^if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P16_Remove_Block]^^89^90^91^^^81^93^if  ( o.getClass (  )  != getClass (  )  )  { return false; }^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P3_Replace_Literal]^return true;^90^^^^^81^93^return false;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P2_Replace_Operator]^return  ( _value !=  (  ( BooleanNode )  o ) ._value ) ;^92^^^^^81^93^return  ( _value ==  (  ( BooleanNode )  o ) ._value ) ;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P5_Replace_Variable]^return  ( v ==  (  ( BooleanNode )  o ) ._value ) ;^92^^^^^81^93^return  ( _value ==  (  ( BooleanNode )  o ) ._value ) ;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  
[P8_Replace_Mix]^return  ( _value  &&   (  ( BooleanNode )  o ) ._value ) ;^92^^^^^81^93^return  ( _value ==  (  ( BooleanNode )  o ) ._value ) ;^[CLASS] BooleanNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] BooleanNode  FALSE  TRUE  Object  o  boolean  _value  b  defaultValue  v  