[BugLab_Variable_Misuse]^result =  ( RegularTimePeriod )  constructor.newInstance ( new Object[] {millisecond, DEFAULT_TIME_ZONE} ) ;^91^92^^^^85^98^result =  ( RegularTimePeriod )  constructor.newInstance ( new Object[] {millisecond, zone} ) ;^[CLASS] RegularTimePeriod  [METHOD] createInstance [RETURN_TYPE] RegularTimePeriod   Class c Date millisecond TimeZone zone [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  Constructor  constructor  Class  c  RegularTimePeriod  result  Date  millisecond  Exception  e  
[BugLab_Argument_Swapping]^result =  ( RegularTimePeriod )  zone.newInstance ( new Object[] {millisecond, constructor} ) ;^91^92^^^^85^98^result =  ( RegularTimePeriod )  constructor.newInstance ( new Object[] {millisecond, zone} ) ;^[CLASS] RegularTimePeriod  [METHOD] createInstance [RETURN_TYPE] RegularTimePeriod   Class c Date millisecond TimeZone zone [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  Constructor  constructor  Class  c  RegularTimePeriod  result  Date  millisecond  Exception  e  
[BugLab_Variable_Misuse]^if  ( c.equals ( Year.c )  )  {^109^^^^^108^133^if  ( c.equals ( Year.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Quarter.c )  )  {^112^^^^^108^133^else if  ( c.equals ( Quarter.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Month.c )  )  {^115^^^^^108^133^else if  ( c.equals ( Month.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Day.c )  )  {^118^^^^^108^133^else if  ( c.equals ( Day.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Hour.c )  )  {^121^^^^^108^133^else if  ( c.equals ( Hour.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Minute.c )  )  {^124^^^^^108^133^else if  ( c.equals ( Minute.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Variable_Misuse]^else if  ( c.equals ( Second.c )  )  {^127^^^^^108^133^else if  ( c.equals ( Second.class )  )  {^[CLASS] RegularTimePeriod  [METHOD] downsize [RETURN_TYPE] Class   Class c [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  Class  c  boolean  
[BugLab_Argument_Swapping]^return m2 +  ( m1 - m1 )  / 2;^258^^^^^255^259^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  long  m1  m2  boolean  
[BugLab_Wrong_Operator]^return m1 +  <  ( m2 - m1 )  / 2;^258^^^^^255^259^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  long  m1  m2  boolean  
[BugLab_Wrong_Operator]^return m1 +  ( m2 - m1 )  + 2;^258^^^^^255^259^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  long  m1  m2  boolean  
[BugLab_Wrong_Operator]^return m1 +  ( m2  ||  m1 )  / 2;^258^^^^^255^259^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  long  m1  m2  boolean  
[BugLab_Wrong_Literal]^return m1 +  ( m - m1 )  / ;^258^^^^^255^259^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  long  m1  m2  boolean  
[BugLab_Argument_Swapping]^return m2 +  ( m1 - m1 )  / 2;^272^^^^^269^273^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  long  m1  m2  Calendar  calendar  
[BugLab_Wrong_Operator]^return m1 +  !=  ( m2 - m1 )  / 2;^272^^^^^269^273^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  long  m1  m2  Calendar  calendar  
[BugLab_Wrong_Operator]^return m1 +  ( m2 - m1 )  * 2;^272^^^^^269^273^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  long  m1  m2  Calendar  calendar  
[BugLab_Wrong_Operator]^return m1 +  ( m2  &&  m1 )  / 2;^272^^^^^269^273^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  long  m1  m2  Calendar  calendar  
[BugLab_Wrong_Literal]^return m1 +  ( m3 - m1 )  / 3;^272^^^^^269^273^return m1 +  ( m2 - m1 )  / 2;^[CLASS] RegularTimePeriod  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] TimeZone  DEFAULT_TIME_ZONE  zone  boolean  long  m1  m2  Calendar  calendar  
