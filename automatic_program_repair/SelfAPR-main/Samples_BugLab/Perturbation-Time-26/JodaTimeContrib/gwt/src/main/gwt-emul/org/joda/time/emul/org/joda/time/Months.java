[BugLab_Wrong_Literal]^public static final Months ONE = new Months ( 0 ) ;^46^^^^^41^51^public static final Months ONE = new Months ( 1 ) ;^[CLASS] Months   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Months FOUR = new Months ( 5 ) ;^52^^^^^47^57^public static final Months FOUR = new Months ( 4 ) ;^[CLASS] Months   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Months SIX = new Months ( 7 ) ;^56^^^^^51^61^public static final Months SIX = new Months ( 6 ) ;^[CLASS] Months   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Months TEN = new Months ( 9 ) ;^64^^^^^59^69^public static final Months TEN = new Months ( 10 ) ;^[CLASS] Months   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Months TWELVE = new Months ( 11 ) ;^68^^^^^63^73^public static final Months TWELVE = new Months ( 12 ) ;^[CLASS] Months   [VARIABLES] 
[BugLab_Variable_Misuse]^return SEVEN;^91^^^^^76^106^return ZERO;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SEVEN;^95^^^^^80^110^return TWO;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SIX;^97^^^^^82^112^return THREE;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SIX;^99^^^^^84^114^return FOUR;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return TWELVE;^103^^^^^88^118^return SIX;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return TEN;^105^^^^^90^120^return SEVEN;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SIX;^107^^^^^92^122^return EIGHT;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return THREE;^109^^^^^94^124^return NINE;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return ZERO;^111^^^^^96^126^return TEN;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SEVEN;^113^^^^^98^128^return ELEVEN;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SEVEN;^115^^^^^100^130^return TWELVE;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return SEVEN;^117^^^^^102^132^return MAX_VALUE;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^return ZERO;^119^^^^^104^134^return MIN_VALUE;^[CLASS] Months  [METHOD] months [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, DurationFieldType.months (  )  ) ;^137^^^^^136^139^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.months (  )  ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, DurationFieldType.months (  )  ) ;^137^^^^^136^139^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.months (  )  ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, DurationFieldType.months (  )  ) ;^137^^^^^136^139^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.months (  )  ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  
[BugLab_Variable_Misuse]^if  ( end instanceof LocalDate && end instanceof LocalDate )    {^154^^^^^153^162^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^if  ( start instanceof LocalDate && start instanceof LocalDate )    {^154^^^^^153^162^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate || end instanceof LocalDate )    {^154^^^^^153^162^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Wrong_Operator]^if  ( start  !=  LocalDate && end instanceof LocalDate )    {^154^^^^^153^162^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate && end  !=  LocalDate )    {^154^^^^^153^162^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^return Months.months ( amount ) ;^158^^^^^153^162^return Months.months ( months ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^Chronology chrono = DateTimeUtils.getChronology ( end.getChronology (  )  ) ;^155^^^^^153^162^Chronology chrono = DateTimeUtils.getChronology ( start.getChronology (  )  ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, ZERO ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, ZERO ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, THREE ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( ZERO, end, start ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, ZERO ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( start, ZERO, end ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, ONE ) ;^160^^^^^153^162^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Variable_Misuse]^return Months.months ( months ) ;^161^^^^^153^162^return Months.months ( amount ) ;^[CLASS] Months  [METHOD] monthsBetween [RETURN_TYPE] Months   ReadablePartial start ReadablePartial end [VARIABLES] ReadablePartial  end  start  Chronology  chrono  boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  amount  months  
[BugLab_Wrong_Operator]^if  ( interval != null )    {^174^^^^^173^179^if  ( interval == null )    {^[CLASS] Months  [METHOD] monthsIn [RETURN_TYPE] Months   ReadableInterval interval [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  ReadableInterval  interval  int  amount  
[BugLab_Wrong_Operator]^if  ( months != 0 )  {^262^^^^^261^266^if  ( months == 0 )  {^[CLASS] Months  [METHOD] plus [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Wrong_Literal]^if  ( months == months )  {^262^^^^^261^266^if  ( months == 0 )  {^[CLASS] Months  [METHOD] plus [RETURN_TYPE] Months   int months [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  long  serialVersionUID  int  months  
[BugLab_Wrong_Operator]^if  ( months != null )  {^278^^^^^277^282^if  ( months == null )  {^[CLASS] Months  [METHOD] plus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return plus ( ZERO.getValue (  )  ) ;^281^^^^^277^282^return plus ( months.getValue (  )  ) ;^[CLASS] Months  [METHOD] plus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return plus ( SIX.getValue (  )  ) ;^281^^^^^277^282^return plus ( months.getValue (  )  ) ;^[CLASS] Months  [METHOD] plus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^308^^^^^307^312^if  ( months == null )  {^[CLASS] Months  [METHOD] minus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( months != null )  {^308^^^^^307^312^if  ( months == null )  {^[CLASS] Months  [METHOD] minus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return minus ( TWELVE.getValue (  )  ) ;^311^^^^^307^312^return minus ( months.getValue (  )  ) ;^[CLASS] Months  [METHOD] minus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return minus ( SEVEN.getValue (  )  ) ;^311^^^^^307^312^return minus ( months.getValue (  )  ) ;^[CLASS] Months  [METHOD] minus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return minus ( TEN.getValue (  )  ) ;^311^^^^^307^312^return minus ( months.getValue (  )  ) ;^[CLASS] Months  [METHOD] minus [RETURN_TYPE] Months   Months months [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( divisor != 1 )  {^339^^^^^338^343^if  ( divisor == 1 )  {^[CLASS] Months  [METHOD] dividedBy [RETURN_TYPE] Months   int divisor [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  int  divisor  
[BugLab_Wrong_Operator]^return Months.months ( getValue (  )  * divisor ) ;^342^^^^^338^343^return Months.months ( getValue (  )  / divisor ) ;^[CLASS] Months  [METHOD] dividedBy [RETURN_TYPE] Months   int divisor [VARIABLES] boolean  Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  long  serialVersionUID  int  divisor  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^364^^^^^363^368^if  ( other == null )  {^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( other != null )  {^364^^^^^363^368^if  ( other == null )  {^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  < 0;^365^^^^^363^368^return getValue (  )  > 0;^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  >= 0;^365^^^^^363^368^return getValue (  )  > 0;^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  > SEVEN.getValue (  ) ;^367^^^^^363^368^return getValue (  )  > other.getValue (  ) ;^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  >= other.getValue (  ) ;^367^^^^^363^368^return getValue (  )  > other.getValue (  ) ;^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  > TWELVE.getValue (  ) ;^367^^^^^363^368^return getValue (  )  > other.getValue (  ) ;^[CLASS] Months  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( SEVEN == null )  {^377^^^^^376^381^if  ( other == null )  {^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( other != null )  {^377^^^^^376^381^if  ( other == null )  {^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  <= 0;^378^^^^^376^381^return getValue (  )  < 0;^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  == 0;^378^^^^^376^381^return getValue (  )  < 0;^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  < SEVEN.getValue (  ) ;^380^^^^^376^381^return getValue (  )  < other.getValue (  ) ;^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  <= other.getValue (  ) ;^380^^^^^376^381^return getValue (  )  < other.getValue (  ) ;^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  < TEN.getValue (  ) ;^380^^^^^376^381^return getValue (  )  < other.getValue (  ) ;^[CLASS] Months  [METHOD] isLessThan [RETURN_TYPE] boolean   Months other [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  ||  )  )  + "M";^392^^^^^391^393^return "P" + String.valueOf ( getValue (  )  )  + "M";^[CLASS] Months  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  &  )  )  + "M";^392^^^^^391^393^return "P" + String.valueOf ( getValue (  )  )  + "M";^[CLASS] Months  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Months  EIGHT  ELEVEN  FIVE  FOUR  MAX_VALUE  MIN_VALUE  NINE  ONE  SEVEN  SIX  TEN  THREE  TWELVE  TWO  ZERO  months  other  long  serialVersionUID  boolean  