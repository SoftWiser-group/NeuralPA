[BugLab_Wrong_Literal]^public static final Days ZERO = new Days ( 1 ) ;^45^^^^^40^50^public static final Days ZERO = new Days ( 0 ) ;^[CLASS] Days   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Days ONE = new Days ( 2 ) ;^47^^^^^42^52^public static final Days ONE = new Days ( 1 ) ;^[CLASS] Days   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Days FOUR = new Days ( null ) ;^53^^^^^48^58^public static final Days FOUR = new Days ( 4 ) ;^[CLASS] Days   [VARIABLES] 
[BugLab_Variable_Misuse]^return TWO;^82^^^^^79^104^return ZERO;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return SIX;^84^^^^^79^104^return ONE;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return SEVEN;^86^^^^^79^104^return TWO;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return ZERO;^88^^^^^79^104^return THREE;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return ZERO;^90^^^^^79^104^return FOUR;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return ZERO;^94^^^^^79^104^return SIX;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return THREE;^96^^^^^79^104^return SEVEN;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return SIX;^98^^^^^79^104^return MAX_VALUE;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return ZERO;^100^^^^^79^104^return MIN_VALUE;^[CLASS] Days  [METHOD] days [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, DurationFieldType.days (  )  ) ;^118^^^^^117^120^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.days (  )  ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  amount  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, DurationFieldType.days (  )  ) ;^118^^^^^117^120^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.days (  )  ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  PeriodFormatter  PARSER  long  serialVersionUID  int  amount  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^if  ( end instanceof LocalDate && end instanceof LocalDate )    {^135^^^^^134^143^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate || end instanceof LocalDate )    {^135^^^^^134^143^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Wrong_Operator]^if  ( start  >>  LocalDate && end instanceof LocalDate )    {^135^^^^^134^143^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate && end  |  LocalDate )    {^135^^^^^134^143^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^return Days.days ( amount ) ;^139^^^^^134^143^return Days.days ( days ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^Chronology chrono = DateTimeUtils.getChronology ( end.getChronology (  )  ) ;^136^^^^^134^143^Chronology chrono = DateTimeUtils.getChronology ( start.getChronology (  )  ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, ZERO ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, ZERO ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, TWO ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, ZERO ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( start, ZERO, end ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( ZERO, end, start ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, SEVEN ) ;^141^^^^^134^143^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Variable_Misuse]^return Days.days ( days ) ;^142^^^^^134^143^return Days.days ( amount ) ;^[CLASS] Days  [METHOD] daysBetween [RETURN_TYPE] Days   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  boolean  PeriodFormatter  PARSER  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  Chronology  chrono  long  serialVersionUID  int  amount  days  
[BugLab_Wrong_Operator]^if  ( interval != null )    {^155^^^^^154^160^if  ( interval == null )    {^[CLASS] Days  [METHOD] daysIn [RETURN_TYPE] Days   ReadableInterval interval [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  ReadableInterval  interval  int  amount  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Operator]^if  ( periodStr != null )  {^200^^^^^199^205^if  ( periodStr == null )  {^[CLASS] Days  [METHOD] parseDays [RETURN_TYPE] Days   String periodStr [VARIABLES] Period  p  String  periodStr  boolean  PeriodFormatter  PARSER  long  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Argument_Swapping]^Period p = periodStr.parsePeriod ( PARSER ) ;^203^^^^^199^205^Period p = PARSER.parsePeriod ( periodStr ) ;^[CLASS] Days  [METHOD] parseDays [RETURN_TYPE] Days   String periodStr [VARIABLES] Period  p  String  periodStr  boolean  PeriodFormatter  PARSER  long  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return Weeks.weeks ( getValue (  )  / THREE ) ;^261^^^^^260^262^return Weeks.weeks ( getValue (  )  / DateTimeConstants.DAYS_PER_WEEK ) ;^[CLASS] Days  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Operator]^return Weeks.weeks ( getValue (  )  + DateTimeConstants.DAYS_PER_WEEK ) ;^261^^^^^260^262^return Weeks.weeks ( getValue (  )  / DateTimeConstants.DAYS_PER_WEEK ) ;^[CLASS] Days  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return Weeks.weeks ( getValue (  )  / SEVEN ) ;^261^^^^^260^262^return Weeks.weeks ( getValue (  )  / DateTimeConstants.DAYS_PER_WEEK ) ;^[CLASS] Days  [METHOD] toStandardWeeks [RETURN_TYPE] Weeks   [VARIABLES] boolean  PeriodFormatter  PARSER  long  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return new Duration ( serialVersionUID * DateTimeConstants.MILLIS_PER_DAY ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return new Duration ( days * ZERO ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Argument_Swapping]^return new Duration ( DateTimeConstants.MILLIS_PER_DAY * days ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Operator]^return new Duration ( days / DateTimeConstants.MILLIS_PER_DAY ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^return new Duration ( days * SIX ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Operator]^return new Duration ( days - DateTimeConstants.MILLIS_PER_DAY ) ;^333^^^^^331^334^return new Duration ( days * DateTimeConstants.MILLIS_PER_DAY ) ;^[CLASS] Days  [METHOD] toStandardDuration [RETURN_TYPE] Duration   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Operator]^if  ( days != 0 )  {^357^^^^^356^361^if  ( days == 0 )  {^[CLASS] Days  [METHOD] plus [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Wrong_Literal]^if  ( days == 1 )  {^357^^^^^356^361^if  ( days == 0 )  {^[CLASS] Days  [METHOD] plus [RETURN_TYPE] Days   int days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  int  days  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  
[BugLab_Variable_Misuse]^if  ( ZERO == null )  {^373^^^^^372^377^if  ( days == null )  {^[CLASS] Days  [METHOD] plus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Wrong_Operator]^if  ( days != null )  {^373^^^^^372^377^if  ( days == null )  {^[CLASS] Days  [METHOD] plus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Variable_Misuse]^return plus ( SEVEN.getValue (  )  ) ;^376^^^^^372^377^return plus ( days.getValue (  )  ) ;^[CLASS] Days  [METHOD] plus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Variable_Misuse]^if  ( SIX == null )  {^403^^^^^402^407^if  ( days == null )  {^[CLASS] Days  [METHOD] minus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Wrong_Operator]^if  ( days != null )  {^403^^^^^402^407^if  ( days == null )  {^[CLASS] Days  [METHOD] minus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Variable_Misuse]^return minus ( THREE.getValue (  )  ) ;^406^^^^^402^407^return minus ( days.getValue (  )  ) ;^[CLASS] Days  [METHOD] minus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Variable_Misuse]^return minus ( SEVEN.getValue (  )  ) ;^406^^^^^402^407^return minus ( days.getValue (  )  ) ;^[CLASS] Days  [METHOD] minus [RETURN_TYPE] Days   Days days [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Wrong_Operator]^if  ( divisor < 1 )  {^434^^^^^433^438^if  ( divisor == 1 )  {^[CLASS] Days  [METHOD] dividedBy [RETURN_TYPE] Days   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  int  divisor  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Wrong_Operator]^return Days.days ( getValue (  )  + divisor ) ;^437^^^^^433^438^return Days.days ( getValue (  )  / divisor ) ;^[CLASS] Days  [METHOD] dividedBy [RETURN_TYPE] Days   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  int  divisor  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Wrong_Operator]^return Days.days ( getValue (  )  * divisor ) ;^437^^^^^433^438^return Days.days ( getValue (  )  / divisor ) ;^[CLASS] Days  [METHOD] dividedBy [RETURN_TYPE] Days   int divisor [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  int  divisor  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  
[BugLab_Variable_Misuse]^if  ( TWO == null )  {^459^^^^^458^463^if  ( other == null )  {^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^if  ( other != null )  {^459^^^^^458^463^if  ( other == null )  {^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  >= 0;^460^^^^^458^463^return getValue (  )  > 0;^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Literal]^return getValue (  )  > -1;^460^^^^^458^463^return getValue (  )  > 0;^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  < 0;^460^^^^^458^463^return getValue (  )  > 0;^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Variable_Misuse]^return getValue (  )  > SIX.getValue (  ) ;^462^^^^^458^463^return getValue (  )  > other.getValue (  ) ;^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  >= other.getValue (  ) ;^462^^^^^458^463^return getValue (  )  > other.getValue (  ) ;^[CLASS] Days  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^if  ( other != null )  {^472^^^^^471^476^if  ( other == null )  {^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  <= 0;^473^^^^^471^476^return getValue (  )  < 0;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Literal]^return getValue (  )  < 1;^473^^^^^471^476^return getValue (  )  < 0;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  > 0;^473^^^^^471^476^return getValue (  )  < 0;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Variable_Misuse]^return getValue (  )  < SIX.getValue (  ) ;^475^^^^^471^476^return getValue (  )  < other.getValue (  ) ;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return getValue (  )  <= other.getValue (  ) ;^475^^^^^471^476^return getValue (  )  < other.getValue (  ) ;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Variable_Misuse]^return getValue (  )  < days.getValue (  ) ;^475^^^^^471^476^return getValue (  )  < other.getValue (  ) ;^[CLASS] Days  [METHOD] isLessThan [RETURN_TYPE] boolean   Days other [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  &  )  )  + "D";^488^^^^^487^489^return "P" + String.valueOf ( getValue (  )  )  + "D";^[CLASS] Days  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  <=  )  )  + "D";^488^^^^^487^489^return "P" + String.valueOf ( getValue (  )  )  + "D";^[CLASS] Days  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  PeriodFormatter  PARSER  long  days  serialVersionUID  Days  FIVE  FOUR  MAX_VALUE  MIN_VALUE  ONE  SEVEN  SIX  THREE  TWO  ZERO  days  other  
