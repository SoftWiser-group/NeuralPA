[BugLab_Wrong_Literal]^private static final int FAST_CACHE_SIZE = 63;^57^^^^^52^62^private static final int FAST_CACHE_SIZE = 64;^[CLASS] ISOChronology Stub   [VARIABLES] 
[BugLab_Variable_Misuse]^iZone = iZone;^217^^^^^216^218^iZone = zone;^[CLASS] ISOChronology Stub  [METHOD] <init> [RETURN_TYPE] DateTimeZone)   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( iZone == null )  {^96^^^^^95^113^if  ( zone == null )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( zone != null )  {^96^^^^^95^113^if  ( zone == null )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^int index = System.identityHashCode ( iZone )  &  ( FAST_CACHE_SIZE - 1 ) ;^99^^^^^95^113^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - 1 ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^int index = System.identityHashCode ( FAST_CACHE_SIZE )  &  ( zone - 1 ) ;^99^^^^^95^113^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - 1 ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^int index = System.identityHashCode ( zone )   <<   ( FAST_CACHE_SIZE - 1 ) ;^99^^^^^95^113^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - 1 ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE  <=  1 ) ;^99^^^^^95^113^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - 1 ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Literal]^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - index ) ;^99^^^^^95^113^int index = System.identityHashCode ( zone )  &  ( FAST_CACHE_SIZE - 1 ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( INSTANCE_UTC != null && chrono.getZone (  )  == zone )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( chrono != null && chrono.getZone (  )  == iZone )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^if  ( zone != null && chrono.getZone (  )  == chrono )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( chrono != null || chrono.getZone (  )  == zone )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( chrono == null && chrono.getZone (  )  == zone )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( chrono != null && chrono.getZone (  )  != zone )  {^101^^^^^95^113^if  ( chrono != null && chrono.getZone (  )  == zone )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^return INSTANCE_UTC;^102^^^^^95^113^return chrono;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( chrono != null )  {^106^^^^^95^113^if  ( chrono == null )  {^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^chrono = new ISOChronology ( ZonedChronology.getInstance ( INSTANCE_UTC, iZone )  ) ;^107^^^^^95^113^chrono = new ISOChronology ( ZonedChronology.getInstance ( INSTANCE_UTC, zone )  ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^chrono = new ISOChronology ( ZonedChronology.getInstance ( zone, INSTANCE_UTC )  ) ;^107^^^^^95^113^chrono = new ISOChronology ( ZonedChronology.getInstance ( INSTANCE_UTC, zone )  ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^chrono = new ISOChronology ( ZonedChronology.getInstance ( chrono, zone )  ) ;^107^^^^^95^113^chrono = new ISOChronology ( ZonedChronology.getInstance ( INSTANCE_UTC, zone )  ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^cCache.put ( iZone, chrono ) ;^108^^^^^95^113^cCache.put ( zone, chrono ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^cCache.put ( zone, INSTANCE_UTC ) ;^108^^^^^95^113^cCache.put ( zone, chrono ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^cCache.put ( chrono, zone ) ;^108^^^^^95^113^cCache.put ( zone, chrono ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^chrono = cCache.get ( iZone ) ;^105^^^^^95^113^chrono = cCache.get ( zone ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^chrono = zone.get ( cCache ) ;^105^^^^^95^113^chrono = cCache.get ( zone ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^chrono = 3.get ( zone ) ;^105^^^^^95^113^chrono = cCache.get ( zone ) ;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^cFastCache[index] = INSTANCE_UTC;^111^^^^^95^113^cFastCache[index] = chrono;^[CLASS] ISOChronology Stub  [METHOD] getInstance [RETURN_TYPE] ISOChronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^return chrono;^133^^^^^132^134^return INSTANCE_UTC;^[CLASS] ISOChronology Stub  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( iZone == null )  {^143^^^^^142^150^if  ( zone == null )  {^[CLASS] ISOChronology Stub  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( zone != null )  {^143^^^^^142^150^if  ( zone == null )  {^[CLASS] ISOChronology Stub  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( iZone == getZone (  )  )  {^146^^^^^142^150^if  ( zone == getZone (  )  )  {^[CLASS] ISOChronology Stub  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( zone != getZone (  )  )  {^146^^^^^142^150^if  ( zone == getZone (  )  )  {^[CLASS] ISOChronology Stub  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^return getInstance ( iZone ) ;^149^^^^^142^150^return getInstance ( zone ) ;^[CLASS] ISOChronology Stub  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^if  ( iZone != null )  {^162^^^^^159^166^if  ( zone != null )  {^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( zone == null )  {^162^^^^^159^166^if  ( zone != null )  {^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Argument_Swapping]^str = zone + '[' + str.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str + '[' + zone.getID (  >>  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str  <<  '['  <<  zone.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str  >=  '[' + zone.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^str = str + '[' + iZone.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str + '[' + zone.getID (  |  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str  ==  '['  ==  zone.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^str = str  &&  '[' + zone.getID (  )  + ']';^163^^^^^159^166^str = str + '[' + zone.getID (  )  + ']';^[CLASS] ISOChronology Stub  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  String  str  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^if  ( getBase (  ) .getZone (  )  <= DateTimeZone.UTC )  {^169^^^^^168^180^if  ( getBase (  ) .getZone (  )  == DateTimeZone.UTC )  {^[CLASS] ISOChronology Stub  [METHOD] assemble [RETURN_TYPE] void   Fields fields [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  Fields  fields  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^fields.centuryOfEra = new DividedDateTimeField ( chrono, DateTimeFieldType.centuryOfEra (  ) , 100 ) ;^171^172^^^^168^180^fields.centuryOfEra = new DividedDateTimeField ( ISOYearOfEraDateTimeField.INSTANCE, DateTimeFieldType.centuryOfEra (  ) , 100 ) ;^[CLASS] ISOChronology Stub  [METHOD] assemble [RETURN_TYPE] void   Fields fields [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  Fields  fields  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Literal]^fields.centuryOfEra = new DividedDateTimeField ( ISOYearOfEraDateTimeField.INSTANCE, DateTimeFieldType.centuryOfEra (  ) , 101 ) ;^171^172^^^^168^180^fields.centuryOfEra = new DividedDateTimeField ( ISOYearOfEraDateTimeField.INSTANCE, DateTimeFieldType.centuryOfEra (  ) , 100 ) ;^[CLASS] ISOChronology Stub  [METHOD] assemble [RETURN_TYPE] void   Fields fields [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  Fields  fields  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Literal]^fields.centuryOfEra = new DividedDateTimeField ( ISOYearOfEraDateTimeField.INSTANCE, DateTimeFieldType.centuryOfEra (  ) , FAST_CACHE_SIZE ) ;^171^172^^^^168^180^fields.centuryOfEra = new DividedDateTimeField ( ISOYearOfEraDateTimeField.INSTANCE, DateTimeFieldType.centuryOfEra (  ) , 100 ) ;^[CLASS] ISOChronology Stub  [METHOD] assemble [RETURN_TYPE] void   Fields fields [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  Fields  fields  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^return "ISO".hashCode (   instanceof   )  * 11 + getZone (  ) .hashCode (  ) ;^200^^^^^199^201^return "ISO".hashCode (  )  * 11 + getZone (  ) .hashCode (  ) ;^[CLASS] ISOChronology Stub  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Operator]^return "ISO".hashCode (  )  / 11 + getZone (  ) .hashCode (  ) ;^200^^^^^199^201^return "ISO".hashCode (  )  * 11 + getZone (  ) .hashCode (  ) ;^[CLASS] ISOChronology Stub  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Wrong_Literal]^return "ISO".hashCode (  )  * 10 + getZone (  ) .hashCode (  ) ;^200^^^^^199^201^return "ISO".hashCode (  )  * 11 + getZone (  ) .hashCode (  ) ;^[CLASS] ISOChronology Stub  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^return ISOChronology.getInstance ( zone ) ;^221^^^^^220^222^return ISOChronology.getInstance ( iZone ) ;^[CLASS] ISOChronology Stub  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] ISOChronology  INSTANCE_UTC  chrono  ISOChronology[]  cFastCache  boolean  Map  cCache  long  serialVersionUID  int  FAST_CACHE_SIZE  index  DateTimeZone  iZone  zone  
[BugLab_Variable_Misuse]^iZone = iZone;^217^^^^^216^218^iZone = zone;^[CLASS] Stub  [METHOD] <init> [RETURN_TYPE] DateTimeZone)   DateTimeZone zone [VARIABLES] long  serialVersionUID  DateTimeZone  iZone  zone  boolean  
[BugLab_Variable_Misuse]^return ISOChronology.getInstance ( zone ) ;^221^^^^^220^222^return ISOChronology.getInstance ( iZone ) ;^[CLASS] Stub  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] long  serialVersionUID  DateTimeZone  iZone  zone  boolean  
[BugLab_Variable_Misuse]^out.writeObject ( zone ) ;^225^^^^^224^226^out.writeObject ( iZone ) ;^[CLASS] Stub  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] boolean  ObjectOutputStream  out  long  serialVersionUID  DateTimeZone  iZone  zone  
