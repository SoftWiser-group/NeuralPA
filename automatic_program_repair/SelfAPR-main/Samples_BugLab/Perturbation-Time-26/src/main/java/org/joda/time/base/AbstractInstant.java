[BugLab_Wrong_Operator]^if  ( type != null )  {^89^^^^^88^93^if  ( type == null )  {^[CLASS] AbstractInstant  [METHOD] get [RETURN_TYPE] int   DateTimeFieldType type [VARIABLES] boolean  DateTimeFieldType  type  
[BugLab_Wrong_Operator]^if  ( type != null )  {^103^^^^^102^107^if  ( type == null )  {^[CLASS] AbstractInstant  [METHOD] isSupported [RETURN_TYPE] boolean   DateTimeFieldType type [VARIABLES] boolean  DateTimeFieldType  type  
[BugLab_Wrong_Literal]^return true;^104^^^^^102^107^return false;^[CLASS] AbstractInstant  [METHOD] isSupported [RETURN_TYPE] boolean   DateTimeFieldType type [VARIABLES] boolean  DateTimeFieldType  type  
[BugLab_Wrong_Operator]^if  ( field != null )  {^124^^^^^123^128^if  ( field == null )  {^[CLASS] AbstractInstant  [METHOD] get [RETURN_TYPE] int   DateTimeField field [VARIABLES] boolean  DateTimeField  field  
[BugLab_Argument_Swapping]^chrono = zone.withZone ( chrono ) ;^166^^^^^164^168^chrono = chrono.withZone ( zone ) ;^[CLASS] AbstractInstant  [METHOD] toDateTime [RETURN_TYPE] DateTime   DateTimeZone zone [VARIABLES] boolean  Chronology  chrono  DateTimeZone  zone  
[BugLab_Argument_Swapping]^chrono = zone.withZone ( chrono ) ;^211^^^^^209^213^chrono = chrono.withZone ( zone ) ;^[CLASS] AbstractInstant  [METHOD] toMutableDateTime [RETURN_TYPE] MutableDateTime   DateTimeZone zone [VARIABLES] boolean  Chronology  chrono  DateTimeZone  zone  
[BugLab_Wrong_Operator]^if  ( this != readableInstant )  {^259^^^^^257^269^if  ( this == readableInstant )  {^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Literal]^return false;^260^^^^^257^269^return true;^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Operator]^if  ( readableInstant instanceof ReadableInstant < false )  {^262^^^^^257^269^if  ( readableInstant instanceof ReadableInstant == false )  {^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Operator]^if  ( readableInstant  >  ReadableInstant == false )  {^262^^^^^257^269^if  ( readableInstant instanceof ReadableInstant == false )  {^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Literal]^if  ( readableInstant instanceof ReadableInstant == true )  {^262^^^^^257^269^if  ( readableInstant instanceof ReadableInstant == false )  {^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Literal]^return true;^263^^^^^257^269^return false;^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Operator]^return getMillis (  )  == otherInstant.getMillis (  )  || FieldUtils.equals ( getChronology (  ) , otherInstant.getChronology (  )  ) ;^266^267^268^^^257^269^return getMillis (  )  == otherInstant.getMillis (  )  && FieldUtils.equals ( getChronology (  ) , otherInstant.getChronology (  )  ) ;^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Operator]^return getMillis (  )  <= otherInstant.getMillis (  )  && FieldUtils.equals ( getChronology (  ) , otherInstant.getChronology (  )  ) ;^266^267^268^^^257^269^return getMillis (  )  == otherInstant.getMillis (  )  && FieldUtils.equals ( getChronology (  ) , otherInstant.getChronology (  )  ) ;^[CLASS] AbstractInstant  [METHOD] equals [RETURN_TYPE] boolean   Object readableInstant [VARIABLES] ReadableInstant  otherInstant  boolean  Object  readableInstant  
[BugLab_Wrong_Operator]^return (  <<  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^278^279^280^^^276^281^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^[CLASS] AbstractInstant  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  
[BugLab_Wrong_Operator]^return (  ( int )   ( getMillis (  )  &  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^278^279^280^^^276^281^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^[CLASS] AbstractInstant  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  
[BugLab_Wrong_Operator]^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )   <  32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^278^279^280^^^276^281^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^[CLASS] AbstractInstant  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  
[BugLab_Wrong_Literal]^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 33 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^278^279^280^^^276^281^return (  ( int )   ( getMillis (  )  ^  ( getMillis (  )  >>> 32 )  )  )  + ( getChronology (  ) .hashCode (  )  ) ;^[CLASS] AbstractInstant  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  
[BugLab_Wrong_Operator]^if  ( this < other )  {^296^^^^^295^312^if  ( this == other )  {^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Literal]^return 1;^297^^^^^295^312^return 0;^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Argument_Swapping]^if  ( otherMillis == thisMillis )  {^304^^^^^295^312^if  ( thisMillis == otherMillis )  {^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^if  ( thisMillis != otherMillis )  {^304^^^^^295^312^if  ( thisMillis == otherMillis )  {^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Literal]^return -1;^305^^^^^295^312^return 0;^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Argument_Swapping]^if  ( otherMillis < thisMillis )  {^307^^^^^295^312^if  ( thisMillis < otherMillis )  {^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^if  ( thisMillis <= otherMillis )  {^307^^^^^295^312^if  ( thisMillis < otherMillis )  {^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Literal]^return 0;^310^^^^^295^312^return 1;^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Literal]^return -0;^308^^^^^295^312^return -1;^[CLASS] AbstractInstant  [METHOD] compareTo [RETURN_TYPE] int   ReadableInstant other [VARIABLES] ReadableInstant  other  boolean  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^return  ( getMillis (  )  >= instant ) ;^323^^^^^322^324^return  ( getMillis (  )  > instant ) ;^[CLASS] AbstractInstant  [METHOD] isAfter [RETURN_TYPE] boolean   long instant [VARIABLES] boolean  long  instant  
[BugLab_Wrong_Operator]^return  ( getMillis (  )  <= instant ) ;^357^^^^^356^358^return  ( getMillis (  )  < instant ) ;^[CLASS] AbstractInstant  [METHOD] isBefore [RETURN_TYPE] boolean   long instant [VARIABLES] boolean  long  instant  
[BugLab_Wrong_Operator]^return  ( getMillis (  )  != instant ) ;^391^^^^^390^392^return  ( getMillis (  )  == instant ) ;^[CLASS] AbstractInstant  [METHOD] isEqual [RETURN_TYPE] boolean   long instant [VARIABLES] boolean  long  instant  
[BugLab_Wrong_Operator]^if  ( formatter != null )  {^436^^^^^435^440^if  ( formatter == null )  {^[CLASS] AbstractInstant  [METHOD] toString [RETURN_TYPE] String   DateTimeFormatter formatter [VARIABLES] boolean  DateTimeFormatter  formatter  
