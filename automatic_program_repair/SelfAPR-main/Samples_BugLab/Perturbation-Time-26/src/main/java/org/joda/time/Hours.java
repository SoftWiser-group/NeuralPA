[BugLab_Wrong_Literal]^public static final Hours ZERO = new Hours ( 1 ) ;^45^^^^^40^50^public static final Hours ZERO = new Hours ( 0 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours ONE = new Hours ( 0 ) ;^47^^^^^42^52^public static final Hours ONE = new Hours ( 1 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours TWO = new Hours ( 3 ) ;^49^^^^^44^54^public static final Hours TWO = new Hours ( 2 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours FIVE = new Hours ( 4 ) ;^55^^^^^50^60^public static final Hours FIVE = new Hours ( 5 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours SIX = new Hours ( 7 ) ;^57^^^^^52^62^public static final Hours SIX = new Hours ( 6 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours SEVEN = new Hours ( 6 ) ;^59^^^^^54^64^public static final Hours SEVEN = new Hours ( 7 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Hours EIGHT = new Hours ( 9 ) ;^61^^^^^56^66^public static final Hours EIGHT = new Hours ( 8 ) ;^[CLASS] Hours   [VARIABLES] 
[BugLab_Variable_Misuse]^return ZERO;^86^^^^^81^108^return ONE;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return ZERO;^88^^^^^81^108^return TWO;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return ZERO;^90^^^^^81^108^return THREE;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return THREE;^92^^^^^81^108^return FOUR;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return ONE;^96^^^^^81^108^return SIX;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return ZERO;^98^^^^^81^108^return SEVEN;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return THREE;^100^^^^^81^108^return EIGHT;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return THREE;^102^^^^^81^108^return MAX_VALUE;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^return THREE;^104^^^^^81^108^return MIN_VALUE;^[CLASS] Hours  [METHOD] hours [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, DurationFieldType.hours (  )  ) ;^121^^^^^120^123^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.hours (  )  ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  amount  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, DurationFieldType.hours (  )  ) ;^121^^^^^120^123^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.hours (  )  ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  int  amount  
[BugLab_Variable_Misuse]^if  ( start instanceof LocalTime && start instanceof LocalTime )    {^138^^^^^137^146^if  ( start instanceof LocalTime && end instanceof LocalTime )    {^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalTime || end instanceof LocalTime )    {^138^^^^^137^146^if  ( start instanceof LocalTime && end instanceof LocalTime )    {^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Wrong_Operator]^if  ( start  ==  LocalTime && end instanceof LocalTime )    {^138^^^^^137^146^if  ( start instanceof LocalTime && end instanceof LocalTime )    {^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalTime && end  <=  LocalTime )    {^138^^^^^137^146^if  ( start instanceof LocalTime && end instanceof LocalTime )    {^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^return Hours.hours ( amount ) ;^142^^^^^137^146^return Hours.hours ( hours ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^Chronology chrono = DateTimeUtils.getChronology ( end.getChronology (  )  ) ;^139^^^^^137^146^Chronology chrono = DateTimeUtils.getChronology ( start.getChronology (  )  ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, ZERO ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, ZERO ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, ZERO ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( start, ZERO, end ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, SIX ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( ZERO, end, start ) ;^144^^^^^137^146^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Variable_Misuse]^return Hours.hours ( hours ) ;^145^^^^^137^146^return Hours.hours ( amount ) ;^[CLASS] Hours  [METHOD] hoursBetween [RETURN_TYPE] Hours   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  hours  
[BugLab_Wrong_Operator]^if  ( interval != null )    {^157^^^^^156^162^if  ( interval == null )    {^[CLASS] Hours  [METHOD] hoursIn [RETURN_TYPE] Hours   ReadableInterval interval [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  ReadableInterval  interval  int  amount  
[BugLab_Wrong_Operator]^if  ( periodStr != null )  {^202^^^^^201^207^if  ( periodStr == null )  {^[CLASS] Hours  [METHOD] parseHours [RETURN_TYPE] Hours   String periodStr [VARIABLES] Period  p  String  periodStr  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Argument_Swapping]^Period p = periodStr.parsePeriod ( PARSER ) ;^205^^^^^201^207^Period p = PARSER.parsePeriod ( periodStr ) ;^[CLASS] Hours  [METHOD] parseHours [RETURN_TYPE] Hours   String periodStr [VARIABLES] Period  p  String  periodStr  boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Variable_Misuse]^return Weeks.weeks ( getValue (  )  / SEVEN ) ;^264^^^^^263^265^return Weeks.weeks ( getValue (  )  / DateTimeConstants.HOURS_PER_WEEK ) ;^[CLASS] Hours  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Wrong_Operator]^return Weeks.weeks ( getValue (  )  - DateTimeConstants.HOURS_PER_WEEK ) ;^264^^^^^263^265^return Weeks.weeks ( getValue (  )  / DateTimeConstants.HOURS_PER_WEEK ) ;^[CLASS] Hours  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Wrong_Operator]^return Weeks.weeks ( getValue (  )  + DateTimeConstants.HOURS_PER_WEEK ) ;^264^^^^^263^265^return Weeks.weeks ( getValue (  )  / DateTimeConstants.HOURS_PER_WEEK ) ;^[CLASS] Hours  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Variable_Misuse]^return Days.days ( getValue (  )  / ONE ) ;^280^^^^^279^281^return Days.days ( getValue (  )  / DateTimeConstants.HOURS_PER_DAY ) ;^[CLASS] Hours  [METHOD] toStandardDays [RETURN_TYPE] Days   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Wrong_Operator]^return Days.days ( getValue (  )  - DateTimeConstants.HOURS_PER_DAY ) ;^280^^^^^279^281^return Days.days ( getValue (  )  / DateTimeConstants.HOURS_PER_DAY ) ;^[CLASS] Hours  [METHOD] toStandardDays [RETURN_TYPE] Days   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Wrong_Operator]^return Days.days ( getValue (  )  + DateTimeConstants.HOURS_PER_DAY ) ;^280^^^^^279^281^return Days.days ( getValue (  )  / DateTimeConstants.HOURS_PER_DAY ) ;^[CLASS] Hours  [METHOD] toStandardDays [RETURN_TYPE] Days   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new Duration ( hours * ONE ) ;^332^^^^^330^333^return new Duration ( hours * DateTimeConstants.MILLIS_PER_HOUR ) ;^[CLASS] Hours  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  
[BugLab_Argument_Swapping]^return new Duration ( DateTimeConstants.MILLIS_PER_HOUR * hours ) ;^332^^^^^330^333^return new Duration ( hours * DateTimeConstants.MILLIS_PER_HOUR ) ;^[CLASS] Hours  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return new Duration ( hours + DateTimeConstants.MILLIS_PER_HOUR ) ;^332^^^^^330^333^return new Duration ( hours * DateTimeConstants.MILLIS_PER_HOUR ) ;^[CLASS] Hours  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return new Duration ( serialVersionUID * DateTimeConstants.MILLIS_PER_HOUR ) ;^332^^^^^330^333^return new Duration ( hours * DateTimeConstants.MILLIS_PER_HOUR ) ;^[CLASS] Hours  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( hours <= 0 )  {^356^^^^^355^360^if  ( hours == 0 )  {^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  int  hours  
[BugLab_Wrong_Literal]^if  ( hours == hours )  {^356^^^^^355^360^if  ( hours == 0 )  {^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   int hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  long  hours  serialVersionUID  int  hours  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^372^^^^^371^376^if  ( hours == null )  {^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( hours != null )  {^372^^^^^371^376^if  ( hours == null )  {^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return plus ( THREE.getValue (  )  ) ;^375^^^^^371^376^return plus ( hours.getValue (  )  ) ;^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return plus ( SIX.getValue (  )  ) ;^375^^^^^371^376^return plus ( hours.getValue (  )  ) ;^[CLASS] Hours  [METHOD] plus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^402^^^^^401^406^if  ( hours == null )  {^[CLASS] Hours  [METHOD] minus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( hours != null )  {^402^^^^^401^406^if  ( hours == null )  {^[CLASS] Hours  [METHOD] minus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return minus ( ZERO.getValue (  )  ) ;^405^^^^^401^406^return minus ( hours.getValue (  )  ) ;^[CLASS] Hours  [METHOD] minus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return minus ( TWO.getValue (  )  ) ;^405^^^^^401^406^return minus ( hours.getValue (  )  ) ;^[CLASS] Hours  [METHOD] minus [RETURN_TYPE] Hours   Hours hours [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( divisor != 1 )  {^433^^^^^432^437^if  ( divisor == 1 )  {^[CLASS] Hours  [METHOD] dividedBy [RETURN_TYPE] Hours   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  int  divisor  
[BugLab_Wrong_Literal]^if  ( divisor == divisor )  {^433^^^^^432^437^if  ( divisor == 1 )  {^[CLASS] Hours  [METHOD] dividedBy [RETURN_TYPE] Hours   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  int  divisor  
[BugLab_Wrong_Operator]^return Hours.hours ( getValue (  )  * divisor ) ;^436^^^^^432^437^return Hours.hours ( getValue (  )  / divisor ) ;^[CLASS] Hours  [METHOD] dividedBy [RETURN_TYPE] Hours   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  int  divisor  
[BugLab_Wrong_Operator]^return Hours.hours ( getValue (  )  - divisor ) ;^436^^^^^432^437^return Hours.hours ( getValue (  )  / divisor ) ;^[CLASS] Hours  [METHOD] dividedBy [RETURN_TYPE] Hours   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  long  hours  serialVersionUID  int  divisor  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^458^^^^^457^462^if  ( other == null )  {^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( other != null )  {^458^^^^^457^462^if  ( other == null )  {^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return getValue (  )  >= 0;^459^^^^^457^462^return getValue (  )  > 0;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return getValue (  )  == 0;^459^^^^^457^462^return getValue (  )  > 0;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Literal]^return getValue (  )  > -1;^459^^^^^457^462^return getValue (  )  > 0;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return getValue (  )  > THREE.getValue (  ) ;^461^^^^^457^462^return getValue (  )  > other.getValue (  ) ;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return getValue (  )  == other.getValue (  ) ;^461^^^^^457^462^return getValue (  )  > other.getValue (  ) ;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return getValue (  )  > SEVEN.getValue (  ) ;^461^^^^^457^462^return getValue (  )  > other.getValue (  ) ;^[CLASS] Hours  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^471^^^^^470^475^if  ( other == null )  {^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( other != null )  {^471^^^^^470^475^if  ( other == null )  {^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return getValue (  )  <= 0;^472^^^^^470^475^return getValue (  )  < 0;^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Literal]^return getValue (  )  < 1;^472^^^^^470^475^return getValue (  )  < 0;^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return getValue (  )  <= other.getValue (  ) ;^474^^^^^470^475^return getValue (  )  < other.getValue (  ) ;^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Variable_Misuse]^return getValue (  )  < THREE.getValue (  ) ;^474^^^^^470^475^return getValue (  )  < other.getValue (  ) ;^[CLASS] Hours  [METHOD] isLessThan [RETURN_TYPE] boolean   Hours other [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return "PT" + String.valueOf ( getValue (  <=  )  )  + "H";^487^^^^^486^488^return "PT" + String.valueOf ( getValue (  )  )  + "H";^[CLASS] Hours  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  
[BugLab_Wrong_Operator]^return "PT" + String.valueOf ( getValue (  ^  )  )  + "H";^487^^^^^486^488^return "PT" + String.valueOf ( getValue (  )  )  + "H";^[CLASS] Hours  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  PeriodFormatter  PARSER  Hours  EIGHT  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  hours  other  long  hours  serialVersionUID  