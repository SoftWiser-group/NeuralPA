[BugLab_Variable_Misuse]^return name;^88^^^^^87^89^return this.name;^[CLASS] TimePeriodAnchor  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( this >= obj )  {^101^^^^^99^114^if  ( this == obj )  {^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^102^^^^^99^114^return true;^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( obj  <=  TimePeriodAnchor )  )  {^104^^^^^99^114^if  ( ! ( obj instanceof TimePeriodAnchor )  )  {^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^105^^^^^99^114^return false;^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.name.equals ( START.name )  )  {^109^^^^^99^114^if  ( !this.name.equals ( position.name )  )  {^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.name.equals ( name )  )  {^109^^^^^99^114^if  ( !this.name.equals ( position.name )  )  {^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.name.equals ( position.name.name )  )  {^109^^^^^99^114^if  ( !this.name.equals ( position.name )  )  {^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^110^^^^^99^114^return false;^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^113^^^^^99^114^return true;^[CLASS] TimePeriodAnchor  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  Object  obj  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return name.hashCode (  ) ;^122^^^^^121^123^return this.name.hashCode (  ) ;^[CLASS] TimePeriodAnchor  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( this.equals ( TimePeriodAnchor.position )  )  {^133^^^^^132^143^if  ( this.equals ( TimePeriodAnchor.START )  )  {^[CLASS] TimePeriodAnchor  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^else if  ( this.equals ( TimePeriodAnchor.position )  )  {^139^^^^^132^143^else if  ( this.equals ( TimePeriodAnchor.END )  )  {^[CLASS] TimePeriodAnchor  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^else if  ( this.equals ( TimePeriodAnchor.position )  )  {^136^^^^^132^143^else if  ( this.equals ( TimePeriodAnchor.MIDDLE )  )  {^[CLASS] TimePeriodAnchor  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] TimePeriodAnchor  END  MIDDLE  START  position  String  name  boolean  long  serialVersionUID  
