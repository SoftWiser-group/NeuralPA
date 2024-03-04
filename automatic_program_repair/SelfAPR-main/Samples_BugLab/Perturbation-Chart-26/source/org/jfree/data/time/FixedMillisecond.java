[BugLab_Variable_Misuse]^this ( new Date ( serialVersionUID )  ) ;^84^^^^^83^85^this ( new Date ( millisecond )  ) ;^[CLASS] FixedMillisecond  [METHOD] <init> [RETURN_TYPE] FixedMillisecond(long)   long millisecond [VARIABLES] long  millisecond  serialVersionUID  Date  time  boolean  
[BugLab_Variable_Misuse]^return time;^102^^^^^101^103^return this.time;^[CLASS] FixedMillisecond  [METHOD] getTime [RETURN_TYPE] Date   [VARIABLES] long  millisecond  serialVersionUID  Date  time  boolean  
[BugLab_Variable_Misuse]^long t = time.getTime (  ) ;^123^^^^^121^128^long t = this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^if  ( serialVersionUID != Long.MIN_VALUE )  {^124^^^^^121^128^if  ( t != Long.MIN_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^if  ( t != Long.serialVersionUID )  {^124^^^^^121^128^if  ( t != Long.MIN_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^if  ( t < Long.MIN_VALUE )  {^124^^^^^121^128^if  ( t != Long.MIN_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^result = new FixedMillisecond ( serialVersionUID - 1 ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  ||  1 ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Literal]^result = new FixedMillisecond ( t  ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  ==  1 ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  <=  1 ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  &  1 ) ;^125^^^^^121^128^result = new FixedMillisecond ( t - 1 ) ;^[CLASS] FixedMillisecond  [METHOD] previous [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^long t = time.getTime (  ) ;^137^^^^^135^142^long t = this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^if  ( serialVersionUID != Long.MAX_VALUE )  {^138^^^^^135^142^if  ( t != Long.MAX_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^if  ( t != Long.serialVersionUID )  {^138^^^^^135^142^if  ( t != Long.MAX_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^if  ( t == Long.MAX_VALUE )  {^138^^^^^135^142^if  ( t != Long.MAX_VALUE )  {^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t   instanceof   1 ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Literal]^result = new FixedMillisecond ( t  ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^result = new FixedMillisecond ( serialVersionUID + 1 ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  >=  1 ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  >  1 ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^result = new FixedMillisecond ( t  >>  1 ) ;^139^^^^^135^142^result = new FixedMillisecond ( t + 1 ) ;^[CLASS] FixedMillisecond  [METHOD] next [RETURN_TYPE] RegularTimePeriod   [VARIABLES] RegularTimePeriod  result  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Operator]^if  ( object  <<  FixedMillisecond )  {^152^^^^^151^160^if  ( object instanceof FixedMillisecond )  {^[CLASS] FixedMillisecond  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  FixedMillisecond  m  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Wrong_Literal]^return true;^157^^^^^151^160^return false;^[CLASS] FixedMillisecond  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  FixedMillisecond  m  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^return time.equals ( m.getTime (  )  ) ;^154^^^^^151^160^return this.time.equals ( m.getTime (  )  ) ;^[CLASS] FixedMillisecond  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  FixedMillisecond  m  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Argument_Swapping]^return m.equals ( this.time.getTime (  )  ) ;^154^^^^^151^160^return this.time.equals ( m.getTime (  )  ) ;^[CLASS] FixedMillisecond  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  FixedMillisecond  m  boolean  long  millisecond  serialVersionUID  t  Date  time  
[BugLab_Variable_Misuse]^return time.hashCode (  ) ;^168^^^^^167^169^return this.time.hashCode (  ) ;^[CLASS] FixedMillisecond  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  millisecond  serialVersionUID  t  Date  time  boolean  
[BugLab_Wrong_Operator]^if  ( o1  <<  FixedMillisecond )  {^187^^^^^172^202^if  ( o1 instanceof FixedMillisecond )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^if  ( o1  !=  FixedMillisecond )  {^187^^^^^172^202^if  ( o1 instanceof FixedMillisecond )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( oresult instanceof FixedMillisecond )  {^187^^^^^172^202^if  ( o1 instanceof FixedMillisecond )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( o2 instanceof FixedMillisecond )  {^187^^^^^172^202^if  ( o1 instanceof FixedMillisecond )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^else if  ( o1  >>  RegularTimePeriod )  {^205^^^^^190^220^else if  ( o1 instanceof RegularTimePeriod )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^else if  ( o instanceof RegularTimePeriod )  {^205^^^^^190^220^else if  ( o1 instanceof RegularTimePeriod )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = result;^207^^^^^192^222^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = 1;^207^^^^^192^222^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^if  ( t > 0 )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^if  ( difference == 0 )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference > 1 )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^if  ( t < 0 )  {^194^^^^^190^200^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^if  ( difference <= 0 )  {^194^^^^^190^200^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference < -1 )  {^194^^^^^190^200^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -;^195^^^^^190^200^result = -1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = result;^191^^^^^176^206^result = 1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^if  ( t < 0 )  {^194^^^^^179^209^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^if  ( difference <= 0 )  {^194^^^^^179^209^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference < 1 )  {^194^^^^^179^209^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = result;^198^^^^^194^199^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^difference = time.getTime (  )  - t1.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^difference = this.time.getTime (  )  - time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Argument_Swapping]^difference = t1.getTime (  )  - this.time.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Argument_Swapping]^difference = t1.time.getTime (  )  - this.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^difference = this.time.getTime (  )   <=  t1.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = ;^191^^^^^176^206^result = 1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -result;^195^^^^^180^210^result = -1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Argument_Swapping]^difference = this.time.getTime (  )  - t1.time.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Argument_Swapping]^difference = this.time.getTime (  )  - t1.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^if  ( difference >= 0 )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference > result )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference >  )  {^190^^^^^175^205^if  ( difference > 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = result;^198^^^^^190^200^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -result;^195^^^^^190^200^result = -1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -2;^195^^^^^190^200^result = -1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference < -1 )  {^194^^^^^179^209^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^if  ( difference < result )  {^194^^^^^179^209^if  ( difference < 0 )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = 1;^198^^^^^194^199^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^else if  ( o1  >=  RegularTimePeriod )  {^205^^^^^190^220^else if  ( o1 instanceof RegularTimePeriod )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^else if  ( oresult instanceof RegularTimePeriod )  {^205^^^^^190^220^else if  ( o1 instanceof RegularTimePeriod )  {^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Operator]^difference = this.time.getTime (  )   >>  t1.time.getTime (  ) ;^189^^^^^174^204^difference = this.time.getTime (  )  - t1.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -2;^195^^^^^180^210^result = -1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = 1;^198^^^^^183^213^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = -1;^207^^^^^192^222^result = 0;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Wrong_Literal]^result = result;^214^^^^^199^229^result = 1;^[CLASS] FixedMillisecond  [METHOD] compareTo [RETURN_TYPE] int   Object o1 [VARIABLES] Object  o1  FixedMillisecond  t1  boolean  long  difference  millisecond  serialVersionUID  t  Date  time  int  result  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^227^^^^^226^228^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getFirstMillisecond [RETURN_TYPE] long   [VARIABLES] long  difference  millisecond  serialVersionUID  t  Date  time  boolean  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^239^^^^^238^240^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getFirstMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] boolean  long  difference  millisecond  serialVersionUID  t  Date  time  Calendar  calendar  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^248^^^^^247^249^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getLastMillisecond [RETURN_TYPE] long   [VARIABLES] long  difference  millisecond  serialVersionUID  t  Date  time  boolean  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^259^^^^^258^260^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getLastMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] boolean  long  difference  millisecond  serialVersionUID  t  Date  time  Calendar  calendar  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^268^^^^^267^269^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   [VARIABLES] long  difference  millisecond  serialVersionUID  t  Date  time  boolean  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^279^^^^^278^280^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getMiddleMillisecond [RETURN_TYPE] long   Calendar calendar [VARIABLES] boolean  long  difference  millisecond  serialVersionUID  t  Date  time  Calendar  calendar  
[BugLab_Variable_Misuse]^return time.getTime (  ) ;^288^^^^^287^289^return this.time.getTime (  ) ;^[CLASS] FixedMillisecond  [METHOD] getSerialIndex [RETURN_TYPE] long   [VARIABLES] long  difference  millisecond  serialVersionUID  t  Date  time  boolean  
