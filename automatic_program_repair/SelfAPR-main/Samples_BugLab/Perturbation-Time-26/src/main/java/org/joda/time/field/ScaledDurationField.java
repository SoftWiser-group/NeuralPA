[BugLab_Argument_Swapping]^super ( type, field ) ;^47^^^^^46^52^super ( field, type ) ;^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Variable_Misuse]^if  ( iScalar == 0 || scalar == 1 )  {^48^^^^^46^52^if  ( scalar == 0 || scalar == 1 )  {^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Wrong_Operator]^if  ( scalar == 0 && scalar == 1 )  {^48^^^^^46^52^if  ( scalar == 0 || scalar == 1 )  {^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Wrong_Operator]^if  ( scalar <= 0 || scalar == 1 )  {^48^^^^^46^52^if  ( scalar == 0 || scalar == 1 )  {^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Wrong_Operator]^if  ( scalar == 0 || scalar != 1 )  {^48^^^^^46^52^if  ( scalar == 0 || scalar == 1 )  {^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Wrong_Literal]^if  ( scalar == 0 || scalar == 2 )  {^48^^^^^46^52^if  ( scalar == 0 || scalar == 1 )  {^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Variable_Misuse]^iScalar = iScalar;^51^^^^^46^52^iScalar = scalar;^[CLASS] ScaledDurationField  [METHOD] <init> [RETURN_TYPE] DurationFieldType,int)   DurationField field DurationFieldType type int scalar [VARIABLES] boolean  DurationField  field  long  serialVersionUID  int  iScalar  scalar  DurationFieldType  type  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValue ( duration )  / scalar;^55^^^^^54^56^return getWrappedField (  ) .getValue ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValue ( iScalar )  / duration;^55^^^^^54^56^return getWrappedField (  ) .getValue ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getValue ( duration )  - iScalar;^55^^^^^54^56^return getWrappedField (  ) .getValue ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValue ( serialVersionUID )  / iScalar;^55^^^^^54^56^return getWrappedField (  ) .getValue ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValueAsLong ( serialVersionUID )  / iScalar;^59^^^^^58^60^return getWrappedField (  ) .getValueAsLong ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValueAsLong ( iScalar )  / duration;^59^^^^^58^60^return getWrappedField (  ) .getValueAsLong ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getValueAsLong ( duration )  - iScalar;^59^^^^^58^60^return getWrappedField (  ) .getValueAsLong ( duration )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration [VARIABLES] long  duration  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValue ( serialVersionUID, instant )  / iScalar;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValue ( duration, serialVersionUID )  / iScalar;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValue ( duration, instant )  / scalar;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValue ( instant, duration )  / iScalar;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValue ( duration, iScalar )  / instant;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getValue ( duration, instant )  - iScalar;^63^^^^^62^64^return getWrappedField (  ) .getValue ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValueAsLong ( duration, serialVersionUID )  / iScalar;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / scalar;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValueAsLong ( instant, duration )  / iScalar;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getValueAsLong ( iScalar, instant )  / duration;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getValueAsLong ( duration, instant )  * iScalar;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getValueAsLong ( serialVersionUID, instant )  / iScalar;^67^^^^^66^68^return getWrappedField (  ) .getValueAsLong ( duration, instant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] long  duration  instant  serialVersionUID  int  iScalar  scalar  boolean  
[BugLab_Wrong_Operator]^long / scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^71^^^^^70^73^long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( serialVersionUID ) ;^72^^^^^70^73^return getWrappedField (  ) .getMillis ( scaled ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^76^^^^^75^78^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, value ) ;^76^^^^^75^78^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^long scaled = FieldUtils.safeMultiply ( iScalar, value ) ;^76^^^^^75^78^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, scalar ) ;^76^^^^^75^78^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( value ) ;^77^^^^^75^78^return getWrappedField (  ) .getMillis ( scaled ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( serialVersionUID ) ;^77^^^^^75^78^return getWrappedField (  ) .getMillis ( scaled ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^long + scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^81^^^^^80^83^long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( value, instant ) ;^82^^^^^80^83^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( scaled, value ) ;^82^^^^^80^83^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getMillis ( instant, scaled ) ;^82^^^^^80^83^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   int value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^86^^^^^85^88^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, value ) ;^86^^^^^85^88^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^long scaled = FieldUtils.safeMultiply ( iScalar, value ) ;^86^^^^^85^88^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( scaled, iScalar ) ;^86^^^^^85^88^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, scalar ) ;^86^^^^^85^88^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( value, instant ) ;^87^^^^^85^88^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getMillis ( scaled, value ) ;^87^^^^^85^88^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getMillis ( instant, scaled ) ;^87^^^^^85^88^return getWrappedField (  ) .getMillis ( scaled, instant ) ;^[CLASS] ScaledDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^long + scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^91^^^^^90^93^long scaled =  (  ( long )  value )  *  (  ( long )  iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .add ( value, scaled ) ;^92^^^^^90^93^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .add ( instant, value ) ;^92^^^^^90^93^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .add ( scaled, instant ) ;^92^^^^^90^93^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( serialVersionUID, iScalar ) ;^96^^^^^95^98^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, value ) ;^96^^^^^95^98^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^long scaled = FieldUtils.safeMultiply ( iScalar, value ) ;^96^^^^^95^98^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^long scaled = FieldUtils.safeMultiply ( value, scalar ) ;^96^^^^^95^98^long scaled = FieldUtils.safeMultiply ( value, iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .add ( value, scaled ) ;^97^^^^^95^98^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .add ( scaled, instant ) ;^97^^^^^95^98^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .add ( instant, value ) ;^97^^^^^95^98^return getWrappedField (  ) .add ( instant, scaled ) ;^[CLASS] ScaledDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] long  duration  instant  scaled  serialVersionUID  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifference ( serialVersionUID, subtrahendInstant )  / iScalar;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifference ( minuendInstant, value )  / iScalar;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / value;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getDifference ( subtrahendInstant, minuendInstant )  / iScalar;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getDifference ( minuendInstant, iScalar )  / subtrahendInstant;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  + iScalar;^101^^^^^100^102^return getWrappedField (  ) .getDifference ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifferenceAsLong ( serialVersionUID, subtrahendInstant )  / iScalar;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, serialVersionUID )  / iScalar;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getDifferenceAsLong ( subtrahendInstant, minuendInstant )  / iScalar;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Argument_Swapping]^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, iScalar )  / subtrahendInstant;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  * iScalar;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getDifferenceAsLong ( value, subtrahendInstant )  / iScalar;^105^^^^^104^106^return getWrappedField (  ) .getDifferenceAsLong ( minuendInstant, subtrahendInstant )  / iScalar;^[CLASS] ScaledDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return getWrappedField (  ) .getUnitMillis (  )  * value;^109^^^^^108^110^return getWrappedField (  ) .getUnitMillis (  )  * iScalar;^[CLASS] ScaledDurationField  [METHOD] getUnitMillis [RETURN_TYPE] long   [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^return + getWrappedField (  ) .getUnitMillis (  )  * iScalar;^109^^^^^108^110^return getWrappedField (  ) .getUnitMillis (  )  * iScalar;^[CLASS] ScaledDurationField  [METHOD] getUnitMillis [RETURN_TYPE] long   [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return value;^119^^^^^118^120^return iScalar;^[CLASS] ScaledDurationField  [METHOD] getScalar [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^if  ( this >= obj )  {^130^^^^^129^139^if  ( this == obj )  {^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^} else if  ( obj  >  ScaledDurationField )  {^132^^^^^129^139^} else if  ( obj instanceof ScaledDurationField )  {^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Variable_Misuse]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( value == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Variable_Misuse]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == value ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Argument_Swapping]^return  ( getWrappedField (  ) .equals ( other.iScalar.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Argument_Swapping]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( other.iScalar == iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  || ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  != other.getType (  )  )  && ( iScalar == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar != other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  <= other.getType (  )  )  && ( iScalar == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar > other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Literal]^return false;^131^^^^^129^139^return true;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^} else if  ( obj  >=  ScaledDurationField )  {^132^^^^^129^139^} else if  ( obj instanceof ScaledDurationField )  {^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Argument_Swapping]^return  ( getWrappedField (  ) .equals ( iScalar.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( other == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Operator]^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  >= other.getType (  )  )  && ( iScalar == other.iScalar ) ;^134^135^136^^^129^139^return  ( getWrappedField (  ) .equals ( other.getWrappedField (  )  )  )  && ( getType (  )  == other.getType (  )  )  && ( iScalar == other.iScalar ) ;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Wrong_Literal]^return true;^138^^^^^129^139^return false;^[CLASS] ScaledDurationField  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  ScaledDurationField  other  boolean  long  duration  instant  minuendInstant  scaled  serialVersionUID  subtrahendInstant  value  int  iScalar  scalar  value  
[BugLab_Variable_Misuse]^long scalar = value;^147^^^^^146^152^long scalar = iScalar;^[CLASS] ScaledDurationField  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  int  hash  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^int hash =  ( int )   ( value ^  ( scalar >>> 32 )  ) ;^148^^^^^146^152^int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^[CLASS] ScaledDurationField  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  int  hash  iScalar  scalar  value  boolean  
[BugLab_Wrong_Operator]^int hash =  ( int )   ( scalar ^  ( scalar  <<  32 )  ) ;^148^^^^^146^152^int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^[CLASS] ScaledDurationField  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  int  hash  iScalar  scalar  value  boolean  
[BugLab_Wrong_Literal]^int hash =  ( int )   ( scalar ^  ( scalar >>> iScalar )  ) ;^148^^^^^146^152^int hash =  ( int )   ( scalar ^  ( scalar >>> 32 )  ) ;^[CLASS] ScaledDurationField  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  int  hash  iScalar  scalar  value  boolean  
[BugLab_Variable_Misuse]^return value;^151^^^^^146^152^return hash;^[CLASS] ScaledDurationField  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  duration  instant  minuendInstant  scalar  scaled  serialVersionUID  subtrahendInstant  value  int  hash  iScalar  scalar  value  boolean  
