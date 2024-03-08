[BugLab_Argument_Swapping]^if  ( otherMillis < thisMillis )  {^97^^^^^92^104^if  ( thisMillis < otherMillis )  {^[CLASS] AbstractDuration  [METHOD] compareTo [RETURN_TYPE] int   ReadableDuration other [VARIABLES] boolean  ReadableDuration  other  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^if  ( thisMillis == otherMillis )  {^97^^^^^92^104^if  ( thisMillis < otherMillis )  {^[CLASS] AbstractDuration  [METHOD] compareTo [RETURN_TYPE] int   ReadableDuration other [VARIABLES] boolean  ReadableDuration  other  long  otherMillis  thisMillis  
[BugLab_Argument_Swapping]^if  ( otherMillis > thisMillis )  {^100^^^^^92^104^if  ( thisMillis > otherMillis )  {^[CLASS] AbstractDuration  [METHOD] compareTo [RETURN_TYPE] int   ReadableDuration other [VARIABLES] boolean  ReadableDuration  other  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^if  ( thisMillis >= otherMillis )  {^100^^^^^92^104^if  ( thisMillis > otherMillis )  {^[CLASS] AbstractDuration  [METHOD] compareTo [RETURN_TYPE] int   ReadableDuration other [VARIABLES] boolean  ReadableDuration  other  long  otherMillis  thisMillis  
[BugLab_Wrong_Literal]^return ;^101^^^^^92^104^return 1;^[CLASS] AbstractDuration  [METHOD] compareTo [RETURN_TYPE] int   ReadableDuration other [VARIABLES] boolean  ReadableDuration  other  long  otherMillis  thisMillis  
[BugLab_Wrong_Operator]^if  ( duration != null )  {^113^^^^^112^117^if  ( duration == null )  {^[CLASS] AbstractDuration  [METHOD] isEqual [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^return compareTo ( duration )  >= 0;^116^^^^^112^117^return compareTo ( duration )  == 0;^[CLASS] AbstractDuration  [METHOD] isEqual [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Literal]^return compareTo ( duration )  == 1;^116^^^^^112^117^return compareTo ( duration )  == 0;^[CLASS] AbstractDuration  [METHOD] isEqual [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^if  ( duration != null )  {^126^^^^^125^130^if  ( duration == null )  {^[CLASS] AbstractDuration  [METHOD] isLongerThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^return compareTo ( duration )  == 0;^129^^^^^125^130^return compareTo ( duration )  > 0;^[CLASS] AbstractDuration  [METHOD] isLongerThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Literal]^return compareTo ( duration )  > -1;^129^^^^^125^130^return compareTo ( duration )  > 0;^[CLASS] AbstractDuration  [METHOD] isLongerThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^if  ( duration != null )  {^139^^^^^138^143^if  ( duration == null )  {^[CLASS] AbstractDuration  [METHOD] isShorterThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^return compareTo ( duration )  <= 0;^142^^^^^138^143^return compareTo ( duration )  < 0;^[CLASS] AbstractDuration  [METHOD] isShorterThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Literal]^return compareTo ( duration )  < -1;^142^^^^^138^143^return compareTo ( duration )  < 0;^[CLASS] AbstractDuration  [METHOD] isShorterThan [RETURN_TYPE] boolean   ReadableDuration duration [VARIABLES] boolean  ReadableDuration  duration  
[BugLab_Wrong_Operator]^if  ( this != duration )  {^154^^^^^153^162^if  ( this == duration )  {^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Literal]^return false;^155^^^^^153^162^return true;^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Operator]^if  ( duration instanceof ReadableDuration <= false )  {^157^^^^^153^162^if  ( duration instanceof ReadableDuration == false )  {^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Operator]^if  ( duration  >>  ReadableDuration == false )  {^157^^^^^153^162^if  ( duration instanceof ReadableDuration == false )  {^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Literal]^if  ( duration instanceof ReadableDuration == true )  {^157^^^^^153^162^if  ( duration instanceof ReadableDuration == false )  {^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Literal]^return true;^158^^^^^153^162^return false;^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Operator]^return  ( getMillis (  )  != other.getMillis (  )  ) ;^161^^^^^153^162^return  ( getMillis (  )  == other.getMillis (  )  ) ;^[CLASS] AbstractDuration  [METHOD] equals [RETURN_TYPE] boolean   Object duration [VARIABLES] boolean  Object  duration  ReadableDuration  other  
[BugLab_Wrong_Operator]^return  ( int )   ( len ^  ( len  >  32 )  ) ;^172^^^^^170^173^return  ( int )   ( len ^  ( len >>> 32 )  ) ;^[CLASS] AbstractDuration  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] boolean  long  len  
[BugLab_Wrong_Operator]^boolean negative =  ( millis <= 0 ) ;^192^^^^^188^204^boolean negative =  ( millis < 0 ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Argument_Swapping]^FormatUtils.appendUnpaddedInteger ( millis, buf ) ;^193^^^^^188^204^FormatUtils.appendUnpaddedInteger ( buf, millis ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Argument_Swapping]^while  ( negative.length (  )  <  ( buf ? 7 : 6 )  )  {^194^^^^^188^204^while  ( buf.length (  )  <  ( negative ? 7 : 6 )  )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^while  ( buf.length (  )  ==  ( negative ? 7 : 6 )  )  {^194^^^^^188^204^while  ( buf.length (  )  <  ( negative ? 7 : 6 )  )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^while  ( buf.length (  )  <  ( negative ? this : 6 )  )  {^194^^^^^188^204^while  ( buf.length (  )  <  ( negative ? 7 : 6 )  )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^buf.insert ( negative ? 3 : 3, "0" ) ;^195^^^^^188^204^buf.insert ( negative ? 3 : 2, "0" ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^buf.insert ( negative ? 3 : 1, "0" ) ;^195^^^^^188^204^buf.insert ( negative ? 3 : 2, "0" ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^if  (  ( millis / 1000 )  * 1000 >= millis )  {^197^^^^^188^204^if  (  ( millis / 1000 )  * 1000 == millis )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^if - (  ( millis / 1000 )  * 1000 == millis )  {^197^^^^^188^204^if  (  ( millis / 1000 )  * 1000 == millis )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^if  (  ( millis + 1000 )  * 1000 == millis )  {^197^^^^^188^204^if  (  ( millis / 1000 )  * 1000 == millis )  {^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^buf.insert ( buf.length (  )   >>  3, "." ) ;^200^^^^^188^204^buf.insert ( buf.length (  )  - 3, "." ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^buf.insert ( buf.length (  )  - this, "." ) ;^200^^^^^188^204^buf.insert ( buf.length (  )  - 3, "." ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^buf.setLength ( buf.length (  )   |  3 ) ;^198^^^^^188^204^buf.setLength ( buf.length (  )  - 3 ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^buf.setLength ( buf.length (  )   ) ;^198^^^^^188^204^buf.setLength ( buf.length (  )  - 3 ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^buf.setLength ( buf.length (  )   >>  3 ) ;^198^^^^^188^204^buf.setLength ( buf.length (  )  - 3 ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Operator]^buf.insert ( buf.length (  )   !=  3, "." ) ;^200^^^^^188^204^buf.insert ( buf.length (  )  - 3, "." ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  
[BugLab_Wrong_Literal]^buf.insert ( buf.length (  )  , "." ) ;^200^^^^^188^204^buf.insert ( buf.length (  )  - 3, "." ) ;^[CLASS] AbstractDuration  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buf  boolean  negative  long  millis  