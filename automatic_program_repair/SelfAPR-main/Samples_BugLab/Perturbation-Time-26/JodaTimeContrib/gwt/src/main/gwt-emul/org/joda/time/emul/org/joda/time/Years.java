[BugLab_Wrong_Literal]^public static final Years ZERO = new Years ( -1 ) ;^44^^^^^39^49^public static final Years ZERO = new Years ( 0 ) ;^[CLASS] Years   [VARIABLES] 
[BugLab_Wrong_Literal]^public static final Years ONE = new Years ( 0 ) ;^46^^^^^41^51^public static final Years ONE = new Years ( 1 ) ;^[CLASS] Years   [VARIABLES] 
[BugLab_Variable_Misuse]^return TWO;^73^^^^^70^87^return ZERO;^[CLASS] Years  [METHOD] years [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^return ZERO;^75^^^^^70^87^return ONE;^[CLASS] Years  [METHOD] years [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^return ZERO;^77^^^^^70^87^return TWO;^[CLASS] Years  [METHOD] years [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^return ZERO;^79^^^^^70^87^return THREE;^[CLASS] Years  [METHOD] years [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^return ZERO;^81^^^^^70^87^return MAX_VALUE;^[CLASS] Years  [METHOD] years [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, DurationFieldType.years (  )  ) ;^101^^^^^100^103^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.years (  )  ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  amount  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, DurationFieldType.years (  )  ) ;^101^^^^^100^103^int amount = BaseSingleFieldPeriod.between ( start, end, DurationFieldType.years (  )  ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadableInstant start ReadableInstant end [VARIABLES] ReadableInstant  end  start  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  amount  
[BugLab_Variable_Misuse]^if  ( end instanceof LocalDate && end instanceof LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^if  ( start instanceof LocalDate && start instanceof LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Argument_Swapping]^if  ( end instanceof LocalDate && start instanceof LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate || end instanceof LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Wrong_Operator]^if  ( start  ==  LocalDate && end instanceof LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Wrong_Operator]^if  ( start instanceof LocalDate && end  <=  LocalDate )    {^118^^^^^117^126^if  ( start instanceof LocalDate && end instanceof LocalDate )    {^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^return Years.years ( amount ) ;^122^^^^^117^126^return Years.years ( years ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^Chronology chrono = DateTimeUtils.getChronology ( end.getChronology (  )  ) ;^119^^^^^117^126^Chronology chrono = DateTimeUtils.getChronology ( start.getChronology (  )  ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( end, end, ZERO ) ;^124^^^^^117^126^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, start, ZERO ) ;^124^^^^^117^126^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^int amount = BaseSingleFieldPeriod.between ( start, end, THREE ) ;^124^^^^^117^126^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( end, start, ZERO ) ;^124^^^^^117^126^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Argument_Swapping]^int amount = BaseSingleFieldPeriod.between ( start, ZERO, end ) ;^124^^^^^117^126^int amount = BaseSingleFieldPeriod.between ( start, end, ZERO ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Variable_Misuse]^return Years.years ( years ) ;^125^^^^^117^126^return Years.years ( amount ) ;^[CLASS] Years  [METHOD] yearsBetween [RETURN_TYPE] Years   ReadablePartial start ReadablePartial end [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  ReadablePartial  end  start  Chronology  chrono  boolean  long  serialVersionUID  int  amount  years  
[BugLab_Wrong_Operator]^if  ( interval != null )    {^138^^^^^137^143^if  ( interval == null )    {^[CLASS] Years  [METHOD] yearsIn [RETURN_TYPE] Years   ReadableInterval interval [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  ReadableInterval  interval  int  amount  
[BugLab_Wrong_Operator]^if  ( years != 0 )  {^226^^^^^225^230^if  ( years == 0 )  {^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Wrong_Literal]^if  ( years == 1 )  {^226^^^^^225^230^if  ( years == 0 )  {^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   int years [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  boolean  long  serialVersionUID  int  years  
[BugLab_Variable_Misuse]^if  ( THREE == null )  {^242^^^^^241^246^if  ( years == null )  {^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Wrong_Operator]^if  ( years != null )  {^242^^^^^241^246^if  ( years == null )  {^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Variable_Misuse]^return plus ( ZERO.getValue (  )  ) ;^245^^^^^241^246^return plus ( years.getValue (  )  ) ;^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Variable_Misuse]^return plus ( THREE.getValue (  )  ) ;^245^^^^^241^246^return plus ( years.getValue (  )  ) ;^[CLASS] Years  [METHOD] plus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Variable_Misuse]^if  ( ZERO == null )  {^272^^^^^271^276^if  ( years == null )  {^[CLASS] Years  [METHOD] minus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Wrong_Operator]^if  ( years != null )  {^272^^^^^271^276^if  ( years == null )  {^[CLASS] Years  [METHOD] minus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Variable_Misuse]^return minus ( THREE.getValue (  )  ) ;^275^^^^^271^276^return minus ( years.getValue (  )  ) ;^[CLASS] Years  [METHOD] minus [RETURN_TYPE] Years   Years years [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  
[BugLab_Wrong_Operator]^if  ( divisor != 1 )  {^303^^^^^302^307^if  ( divisor == 1 )  {^[CLASS] Years  [METHOD] dividedBy [RETURN_TYPE] Years   int divisor [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  long  serialVersionUID  int  divisor  
[BugLab_Wrong_Literal]^if  ( divisor == 2 )  {^303^^^^^302^307^if  ( divisor == 1 )  {^[CLASS] Years  [METHOD] dividedBy [RETURN_TYPE] Years   int divisor [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  long  serialVersionUID  int  divisor  
[BugLab_Wrong_Operator]^return Years.years ( getValue (  )  * divisor ) ;^306^^^^^302^307^return Years.years ( getValue (  )  / divisor ) ;^[CLASS] Years  [METHOD] dividedBy [RETURN_TYPE] Years   int divisor [VARIABLES] Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  years  boolean  long  serialVersionUID  int  divisor  
[BugLab_Wrong_Operator]^if  ( other != null )  {^328^^^^^327^332^if  ( other == null )  {^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  >= 0;^329^^^^^327^332^return getValue (  )  > 0;^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Literal]^return getValue (  )  > -1;^329^^^^^327^332^return getValue (  )  > 0;^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  > TWO.getValue (  ) ;^331^^^^^327^332^return getValue (  )  > other.getValue (  ) ;^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  >= other.getValue (  ) ;^331^^^^^327^332^return getValue (  )  > other.getValue (  ) ;^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  > years.getValue (  ) ;^331^^^^^327^332^return getValue (  )  > other.getValue (  ) ;^[CLASS] Years  [METHOD] isGreaterThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^if  ( other != null )  {^341^^^^^340^345^if  ( other == null )  {^[CLASS] Years  [METHOD] isLessThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  <= 0;^342^^^^^340^345^return getValue (  )  < 0;^[CLASS] Years  [METHOD] isLessThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Literal]^return getValue (  )  < -1;^342^^^^^340^345^return getValue (  )  < 0;^[CLASS] Years  [METHOD] isLessThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return getValue (  )  == other.getValue (  ) ;^344^^^^^340^345^return getValue (  )  < other.getValue (  ) ;^[CLASS] Years  [METHOD] isLessThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Variable_Misuse]^return getValue (  )  < ZERO.getValue (  ) ;^344^^^^^340^345^return getValue (  )  < other.getValue (  ) ;^[CLASS] Years  [METHOD] isLessThan [RETURN_TYPE] boolean   Years other [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  >>  )  )  + "Y";^356^^^^^355^357^return "P" + String.valueOf ( getValue (  )  )  + "Y";^[CLASS] Years  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
[BugLab_Wrong_Operator]^return "P" + String.valueOf ( getValue (  >=  )  )  + "Y";^356^^^^^355^357^return "P" + String.valueOf ( getValue (  )  )  + "Y";^[CLASS] Years  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  Years  MAX_VALUE  MIN_VALUE  ONE  THREE  TWO  ZERO  other  years  boolean  
