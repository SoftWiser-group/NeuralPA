[BugLab_Wrong_Literal]^return false;^62^^^^^61^63^return true;^[CLASS] MillisDurationField  [METHOD] isSupported [RETURN_TYPE] boolean   [VARIABLES] DurationField  INSTANCE  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return false;^71^^^^^70^72^return true;^[CLASS] MillisDurationField  [METHOD] isPrecise [RETURN_TYPE] boolean   [VARIABLES] DurationField  INSTANCE  long  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return 0;^80^^^^^79^81^return 1;^[CLASS] MillisDurationField  [METHOD] getUnitMillis [RETURN_TYPE] long   [VARIABLES] DurationField  INSTANCE  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeToInt ( serialVersionUID ) ;^85^^^^^84^86^return FieldUtils.safeToInt ( duration ) ;^[CLASS] MillisDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration [VARIABLES] DurationField  INSTANCE  long  duration  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^89^^^^^88^90^return duration;^[CLASS] MillisDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration [VARIABLES] DurationField  INSTANCE  long  duration  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeToInt ( serialVersionUID ) ;^93^^^^^92^94^return FieldUtils.safeToInt ( duration ) ;^[CLASS] MillisDurationField  [METHOD] getValue [RETURN_TYPE] int   long duration long instant [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^97^^^^^96^98^return duration;^[CLASS] MillisDurationField  [METHOD] getValueAsLong [RETURN_TYPE] long   long duration long instant [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^105^^^^^104^106^return value;^[CLASS] MillisDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^113^^^^^112^114^return value;^[CLASS] MillisDurationField  [METHOD] getMillis [RETURN_TYPE] long   long value long instant [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeAdd ( serialVersionUID, value ) ;^117^^^^^116^118^return FieldUtils.safeAdd ( instant, value ) ;^[CLASS] MillisDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] boolean  DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  int  value  
[BugLab_Argument_Swapping]^return FieldUtils.safeAdd ( value, instant ) ;^117^^^^^116^118^return FieldUtils.safeAdd ( instant, value ) ;^[CLASS] MillisDurationField  [METHOD] add [RETURN_TYPE] long   long instant int value [VARIABLES] boolean  DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  int  value  
[BugLab_Variable_Misuse]^return FieldUtils.safeAdd ( serialVersionUID, value ) ;^121^^^^^120^122^return FieldUtils.safeAdd ( instant, value ) ;^[CLASS] MillisDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeAdd ( instant, serialVersionUID ) ;^121^^^^^120^122^return FieldUtils.safeAdd ( instant, value ) ;^[CLASS] MillisDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  boolean  
[BugLab_Argument_Swapping]^return FieldUtils.safeAdd ( value, instant ) ;^121^^^^^120^122^return FieldUtils.safeAdd ( instant, value ) ;^[CLASS] MillisDurationField  [METHOD] add [RETURN_TYPE] long   long instant long value [VARIABLES] DurationField  INSTANCE  long  duration  instant  serialVersionUID  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( value, subtrahendInstant )  ) ;^125^^^^^124^126^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant )  ) ;^[CLASS] MillisDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( minuendInstant, value )  ) ;^125^^^^^124^126^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant )  ) ;^[CLASS] MillisDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Argument_Swapping]^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( subtrahendInstant, minuendInstant )  ) ;^125^^^^^124^126^return FieldUtils.safeToInt ( FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant )  ) ;^[CLASS] MillisDurationField  [METHOD] getDifference [RETURN_TYPE] int   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeSubtract ( value, subtrahendInstant ) ;^129^^^^^128^130^return FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant ) ;^[CLASS] MillisDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Variable_Misuse]^return FieldUtils.safeSubtract ( minuendInstant, value ) ;^129^^^^^128^130^return FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant ) ;^[CLASS] MillisDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Argument_Swapping]^return FieldUtils.safeSubtract ( subtrahendInstant, minuendInstant ) ;^129^^^^^128^130^return FieldUtils.safeSubtract ( minuendInstant, subtrahendInstant ) ;^[CLASS] MillisDurationField  [METHOD] getDifferenceAsLong [RETURN_TYPE] long   long minuendInstant long subtrahendInstant [VARIABLES] DurationField  INSTANCE  long  duration  instant  minuendInstant  serialVersionUID  subtrahendInstant  value  boolean  
[BugLab_Variable_Misuse]^long otherMillis = INSTANCE.getUnitMillis (  ) ;^134^^^^^133^145^long otherMillis = otherField.getUnitMillis (  ) ;^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Variable_Misuse]^if  ( subtrahendInstant == otherMillis )  {^137^^^^^133^145^if  ( thisMillis == otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Variable_Misuse]^if  ( thisMillis == value )  {^137^^^^^133^145^if  ( thisMillis == otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Argument_Swapping]^if  ( otherMillis == thisMillis )  {^137^^^^^133^145^if  ( thisMillis == otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Wrong_Operator]^if  ( thisMillis != otherMillis )  {^137^^^^^133^145^if  ( thisMillis == otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Variable_Misuse]^if  ( value < otherMillis )  {^140^^^^^133^145^if  ( thisMillis < otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Variable_Misuse]^if  ( thisMillis < value )  {^140^^^^^133^145^if  ( thisMillis < otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Argument_Swapping]^if  ( otherMillis < thisMillis )  {^140^^^^^133^145^if  ( thisMillis < otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Wrong_Operator]^if  ( thisMillis > otherMillis )  {^140^^^^^133^145^if  ( thisMillis < otherMillis )  {^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Wrong_Literal]^return -2;^141^^^^^133^145^return -1;^[CLASS] MillisDurationField  [METHOD] compareTo [RETURN_TYPE] int   DurationField otherField [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
[BugLab_Variable_Misuse]^return otherField;^160^^^^^159^161^return INSTANCE;^[CLASS] MillisDurationField  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] DurationField  INSTANCE  otherField  long  duration  instant  minuendInstant  otherMillis  serialVersionUID  subtrahendInstant  thisMillis  value  boolean  
