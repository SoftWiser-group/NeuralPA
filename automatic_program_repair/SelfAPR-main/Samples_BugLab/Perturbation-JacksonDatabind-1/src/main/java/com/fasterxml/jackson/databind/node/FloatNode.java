[BugLab_Variable_Misuse]^public FloatNode ( float _value )  { _value = v; }^27^^^^^22^32^public FloatNode ( float v )  { _value = v; }^[CLASS] FloatNode  [METHOD] <init> [RETURN_TYPE] FloatNode(float)   float v [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^public static FloatNode _valuealueOf ( float v )  { return new FloatNode ( v ) ; }^29^^^^^24^34^public static FloatNode valueOf ( float v )  { return new FloatNode ( v ) ; }^[CLASS] FloatNode  [METHOD] valueOf [RETURN_TYPE] FloatNode   float v [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Literal]^public boolean isFloatingPointNumber (  )  { return false; }^49^^^^^44^54^public boolean isFloatingPointNumber (  )  { return true; }^[CLASS] FloatNode  [METHOD] isFloatingPointNumber [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Literal]^public boolean isFloat (  )  { return false; }^52^^^^^47^57^public boolean isFloat (  )  { return true; }^[CLASS] FloatNode  [METHOD] isFloat [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^return  ( v >= Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^55^^^^^54^56^return  ( _value >= Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToInt [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value >= Integer.MIN_VALUE || _value <= Integer.MAX_VALUE ) ;^55^^^^^54^56^return  ( _value >= Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToInt [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value > Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^55^^^^^54^56^return  ( _value >= Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToInt [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value >= Integer.MIN_VALUE && _value < Integer.MAX_VALUE ) ;^55^^^^^54^56^return  ( _value >= Integer.MIN_VALUE && _value <= Integer.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToInt [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^return  ( v >= Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^59^^^^^58^60^return  ( _value >= Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToLong [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value >= Long.MIN_VALUE || _value <= Long.MAX_VALUE ) ;^59^^^^^58^60^return  ( _value >= Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToLong [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value > Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^59^^^^^58^60^return  ( _value >= Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToLong [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Wrong_Operator]^return  ( _value >= Long.MIN_VALUE && _value < Long.MAX_VALUE ) ;^59^^^^^58^60^return  ( _value >= Long.MIN_VALUE && _value <= Long.MAX_VALUE ) ;^[CLASS] FloatNode  [METHOD] canConvertToLong [RETURN_TYPE] boolean   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^return Float.valueOf ( v ) ;^64^^^^^63^65^return Float.valueOf ( _value ) ;^[CLASS] FloatNode  [METHOD] numberValue [RETURN_TYPE] Number   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^public double doubleValue (  )  { return v; }^80^^^^^75^85^public double doubleValue (  )  { return _value; }^[CLASS] FloatNode  [METHOD] doubleValue [RETURN_TYPE] double   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^public BigDecimal decimalValue (  )  { return BigDecimal.valueOf ( v ) ; }^83^^^^^78^88^public BigDecimal decimalValue (  )  { return BigDecimal.valueOf ( _value ) ; }^[CLASS] FloatNode  [METHOD] decimalValue [RETURN_TYPE] BigDecimal   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^return NumberOutput.toString ( v ) ;^92^^^^^91^93^return NumberOutput.toString ( _value ) ;^[CLASS] FloatNode  [METHOD] asText [RETURN_TYPE] String   [VARIABLES] float  _value  v  boolean  
[BugLab_Variable_Misuse]^jg.writeNumber ( v ) ;^99^^^^^96^100^jg.writeNumber ( _value ) ;^[CLASS] FloatNode  [METHOD] serialize [RETURN_TYPE] void   JsonGenerator jg SerializerProvider provider [VARIABLES] JsonGenerator  jg  boolean  float  _value  v  SerializerProvider  provider  
[BugLab_Wrong_Operator]^if  ( o != this )  return true;^105^^^^^103^115^if  ( o == this )  return true;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Literal]^if  ( o == this )  return false;^105^^^^^103^115^if  ( o == this )  return true;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Operator]^if  ( o != null )  return false;^106^^^^^103^115^if  ( o == null )  return false;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Literal]^if  ( o == null )  return true;^106^^^^^103^115^if  ( o == null )  return false;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Operator]^if  ( o.getClass (  )  == getClass (  )  )  {^107^^^^^103^115^if  ( o.getClass (  )  != getClass (  )  )  {^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Literal]^return true;^108^^^^^103^115^return false;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Variable_Misuse]^return Float.compare ( _value, v )  == 0;^114^^^^^103^115^return Float.compare ( _value, otherValue )  == 0;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Variable_Misuse]^return Float.compare ( v, otherValue )  == 0;^114^^^^^103^115^return Float.compare ( _value, otherValue )  == 0;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Argument_Swapping]^return Float.compare ( otherValue, _value )  == 0;^114^^^^^103^115^return Float.compare ( _value, otherValue )  == 0;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Wrong_Operator]^return Float.compare ( _value, otherValue )  >= 0;^114^^^^^103^115^return Float.compare ( _value, otherValue )  == 0;^[CLASS] FloatNode  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] float  _value  otherValue  v  Object  o  boolean  
[BugLab_Variable_Misuse]^return Float.floatToIntBits ( v ) ;^119^^^^^118^120^return Float.floatToIntBits ( _value ) ;^[CLASS] FloatNode  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] float  _value  otherValue  v  boolean  