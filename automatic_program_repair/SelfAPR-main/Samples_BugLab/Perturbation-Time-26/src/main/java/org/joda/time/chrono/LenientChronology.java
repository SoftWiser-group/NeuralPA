[BugLab_Variable_Misuse]^super ( iWithUTC, null ) ;^59^^^^^58^60^super ( base, null ) ;^[CLASS] LenientChronology  [METHOD] <init> [RETURN_TYPE] Chronology)   Chronology base [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^if  ( base != null )  {^45^^^^^44^49^if  ( base == null )  {^[CLASS] LenientChronology  [METHOD] getInstance [RETURN_TYPE] LenientChronology   Chronology base [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Variable_Misuse]^return new LenientChronology ( iWithUTC ) ;^48^^^^^44^49^return new LenientChronology ( base ) ;^[CLASS] LenientChronology  [METHOD] getInstance [RETURN_TYPE] LenientChronology   Chronology base [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Variable_Misuse]^if  ( base == null )  {^63^^^^^62^71^if  ( iWithUTC == null )  {^[CLASS] LenientChronology  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^if  ( iWithUTC != null )  {^63^^^^^62^71^if  ( iWithUTC == null )  {^[CLASS] LenientChronology  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^if  ( getZone (  )  >= DateTimeZone.UTC )  {^64^^^^^62^71^if  ( getZone (  )  == DateTimeZone.UTC )  {^[CLASS] LenientChronology  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^if  ( getZone (  )  != DateTimeZone.UTC )  {^64^^^^^62^71^if  ( getZone (  )  == DateTimeZone.UTC )  {^[CLASS] LenientChronology  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Variable_Misuse]^return base;^70^^^^^62^71^return iWithUTC;^[CLASS] LenientChronology  [METHOD] withUTC [RETURN_TYPE] Chronology   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^if  ( zone != null )  {^74^^^^^73^84^if  ( zone == null )  {^[CLASS] LenientChronology  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] Chronology  base  iWithUTC  boolean  long  serialVersionUID  DateTimeZone  zone  
[BugLab_Wrong_Operator]^if  ( zone >= DateTimeZone.UTC )  {^77^^^^^73^84^if  ( zone == DateTimeZone.UTC )  {^[CLASS] LenientChronology  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] Chronology  base  iWithUTC  boolean  long  serialVersionUID  DateTimeZone  zone  
[BugLab_Wrong_Operator]^if  ( zone != getZone (  )  )  {^80^^^^^73^84^if  ( zone == getZone (  )  )  {^[CLASS] LenientChronology  [METHOD] withZone [RETURN_TYPE] Chronology   DateTimeZone zone [VARIABLES] Chronology  base  iWithUTC  boolean  long  serialVersionUID  DateTimeZone  zone  
[BugLab_Wrong_Operator]^if  ( this <= obj )  {^127^^^^^126^135^if  ( this == obj )  {^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^128^^^^^126^135^return true;^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( obj instanceof LenientChronology != false )  {^130^^^^^126^135^if  ( obj instanceof LenientChronology == false )  {^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( obj  >>  LenientChronology == false )  {^130^^^^^126^135^if  ( obj instanceof LenientChronology == false )  {^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^if  ( obj instanceof LenientChronology == true )  {^130^^^^^126^135^if  ( obj instanceof LenientChronology == false )  {^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^131^^^^^126^135^return false;^[CLASS] LenientChronology  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  Chronology  base  iWithUTC  LenientChronology  chrono  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return 236548278 + getBase (  &  ) .hashCode (  )  * 7;^144^^^^^143^145^return 236548278 + getBase (  ) .hashCode (  )  * 7;^[CLASS] LenientChronology  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^return - 236548278 + getBase (  ) .hashCode (  )  * 7;^144^^^^^143^145^return 236548278 + getBase (  ) .hashCode (  )  * 7;^[CLASS] LenientChronology  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Literal]^return 236548268 + getBase (  ) .hashCode (  )  * 6;^144^^^^^143^145^return 236548278 + getBase (  ) .hashCode (  )  * 7;^[CLASS] LenientChronology  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^return "LenientChronology[" + getBase (  >  ) .toString (  )  + ']';^153^^^^^152^154^return "LenientChronology[" + getBase (  ) .toString (  )  + ']';^[CLASS] LenientChronology  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
[BugLab_Wrong_Operator]^return "LenientChronology["  >  getBase (  ) .toString (  )  + ']';^153^^^^^152^154^return "LenientChronology[" + getBase (  ) .toString (  )  + ']';^[CLASS] LenientChronology  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  Chronology  base  iWithUTC  boolean  
