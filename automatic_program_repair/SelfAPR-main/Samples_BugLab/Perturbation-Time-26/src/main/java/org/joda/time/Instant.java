[BugLab_Variable_Misuse]^iMillis = serialVersionUID;^106^^^^^104^107^iMillis = instant;^[CLASS] Instant  [METHOD] <init> [RETURN_TYPE] Instant(long)   long instant [VARIABLES] long  iMillis  instant  serialVersionUID  boolean  
[BugLab_Argument_Swapping]^iMillis = instant.getInstantMillis ( converter, ISOChronology.getInstanceUTC (  )  ) ;^121^^^^^118^122^iMillis = converter.getInstantMillis ( instant, ISOChronology.getInstanceUTC (  )  ) ;^[CLASS] Instant  [METHOD] <init> [RETURN_TYPE] Object)   Object instant [VARIABLES] Object  instant  InstantConverter  converter  boolean  long  iMillis  instant  serialVersionUID  
[BugLab_Argument_Swapping]^return str.parseDateTime ( formatter ) .toInstant (  ) ;^87^^^^^86^88^return formatter.parseDateTime ( str ) .toInstant (  ) ;^[CLASS] Instant  [METHOD] parse [RETURN_TYPE] Instant   String str DateTimeFormatter formatter [VARIABLES] String  str  boolean  DateTimeFormatter  formatter  long  iMillis  instant  serialVersionUID  
[BugLab_Variable_Misuse]^return  ( serialVersionUID == iMillis ? this : new Instant ( newMillis )  ) ;^144^^^^^143^145^return  ( newMillis == iMillis ? this : new Instant ( newMillis )  ) ;^[CLASS] Instant  [METHOD] withMillis [RETURN_TYPE] Instant   long newMillis [VARIABLES] long  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return  ( newMillis == serialVersionUID ? this : new Instant ( newMillis )  ) ;^144^^^^^143^145^return  ( newMillis == iMillis ? this : new Instant ( newMillis )  ) ;^[CLASS] Instant  [METHOD] withMillis [RETURN_TYPE] Instant   long newMillis [VARIABLES] long  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Argument_Swapping]^return  ( iMillis == newMillis ? this : new Instant ( newMillis )  ) ;^144^^^^^143^145^return  ( newMillis == iMillis ? this : new Instant ( newMillis )  ) ;^[CLASS] Instant  [METHOD] withMillis [RETURN_TYPE] Instant   long newMillis [VARIABLES] long  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return  ( newMillis <= iMillis ? this : new Instant ( newMillis )  ) ;^144^^^^^143^145^return  ( newMillis == iMillis ? this : new Instant ( newMillis )  ) ;^[CLASS] Instant  [METHOD] withMillis [RETURN_TYPE] Instant   long newMillis [VARIABLES] long  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( newMillis == 0 || scalar == 0 )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Argument_Swapping]^if  ( scalar == 0 || durationToAdd == 0 )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Wrong_Operator]^if  ( durationToAdd == 0 && scalar == 0 )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Wrong_Operator]^if  ( durationToAdd <= 0 || scalar == 0 )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Wrong_Operator]^if  ( durationToAdd == 0 || scalar >= 0 )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Wrong_Literal]^if  ( durationToAdd == scalar || scalar == scalar )  {^158^^^^^157^163^if  ( durationToAdd == 0 || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Variable_Misuse]^long instant = getChronology (  ) .add ( getMillis (  ) , serialVersionUID, scalar ) ;^161^^^^^157^163^long instant = getChronology (  ) .add ( getMillis (  ) , durationToAdd, scalar ) ;^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Argument_Swapping]^long instant = getChronology (  ) .add ( getMillis (  ) , scalar, durationToAdd ) ;^161^^^^^157^163^long instant = getChronology (  ) .add ( getMillis (  ) , durationToAdd, scalar ) ;^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Variable_Misuse]^return withMillis ( serialVersionUID ) ;^162^^^^^157^163^return withMillis ( instant ) ;^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   long durationToAdd int scalar [VARIABLES] long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  boolean  
[BugLab_Argument_Swapping]^if  ( scalar == null || durationToAdd == 0 )  {^176^^^^^175^180^if  ( durationToAdd == null || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   ReadableDuration durationToAdd int scalar [VARIABLES] boolean  ReadableDuration  durationToAdd  long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  
[BugLab_Wrong_Operator]^if  ( durationToAdd == null && scalar == 0 )  {^176^^^^^175^180^if  ( durationToAdd == null || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   ReadableDuration durationToAdd int scalar [VARIABLES] boolean  ReadableDuration  durationToAdd  long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  
[BugLab_Wrong_Operator]^if  ( durationToAdd != null || scalar == 0 )  {^176^^^^^175^180^if  ( durationToAdd == null || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   ReadableDuration durationToAdd int scalar [VARIABLES] boolean  ReadableDuration  durationToAdd  long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  
[BugLab_Wrong_Operator]^if  ( durationToAdd == null || scalar != 0 )  {^176^^^^^175^180^if  ( durationToAdd == null || scalar == 0 )  {^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   ReadableDuration durationToAdd int scalar [VARIABLES] boolean  ReadableDuration  durationToAdd  long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  
[BugLab_Argument_Swapping]^return withDurationAdded ( scalar.getMillis (  ) , durationToAdd ) ;^179^^^^^175^180^return withDurationAdded ( durationToAdd.getMillis (  ) , scalar ) ;^[CLASS] Instant  [METHOD] withDurationAdded [RETURN_TYPE] Instant   ReadableDuration durationToAdd int scalar [VARIABLES] boolean  ReadableDuration  durationToAdd  long  durationToAdd  iMillis  instant  newMillis  serialVersionUID  int  scalar  
[BugLab_Variable_Misuse]^return withDurationAdded ( serialVersionUID, 1 ) ;^193^^^^^192^194^return withDurationAdded ( duration, 1 ) ;^[CLASS] Instant  [METHOD] plus [RETURN_TYPE] Instant   long duration [VARIABLES] long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return withDurationAdded ( duration, 0 ) ;^193^^^^^192^194^return withDurationAdded ( duration, 1 ) ;^[CLASS] Instant  [METHOD] plus [RETURN_TYPE] Instant   long duration [VARIABLES] long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return withDurationAdded ( duration, 2 ) ;^206^^^^^205^207^return withDurationAdded ( duration, 1 ) ;^[CLASS] Instant  [METHOD] plus [RETURN_TYPE] Instant   ReadableDuration duration [VARIABLES] ReadableDuration  duration  long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return withDurationAdded ( serialVersionUID, -1 ) ;^220^^^^^219^221^return withDurationAdded ( duration, -1 ) ;^[CLASS] Instant  [METHOD] minus [RETURN_TYPE] Instant   long duration [VARIABLES] long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Wrong_Literal]^return withDurationAdded ( duration, -2 ) ;^233^^^^^232^234^return withDurationAdded ( duration, -1 ) ;^[CLASS] Instant  [METHOD] minus [RETURN_TYPE] Instant   ReadableDuration duration [VARIABLES] ReadableDuration  duration  long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return serialVersionUID;^243^^^^^242^244^return iMillis;^[CLASS] Instant  [METHOD] getMillis [RETURN_TYPE] long   [VARIABLES] long  duration  durationToAdd  iMillis  instant  newMillis  serialVersionUID  boolean  