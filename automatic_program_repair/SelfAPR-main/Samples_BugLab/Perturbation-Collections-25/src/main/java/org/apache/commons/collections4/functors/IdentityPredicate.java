[BugLab_Variable_Misuse]^iValue = iValue;^61^^^^^59^62^iValue = object;^[CLASS] IdentityPredicate  [METHOD] <init> [RETURN_TYPE] IdentityPredicate(T)   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( iValue == null )  {^47^^^^^46^51^if  ( object == null )  {^[CLASS] IdentityPredicate  [METHOD] identityPredicate [RETURN_TYPE] <T>   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^if  ( object != null )  {^47^^^^^46^51^if  ( object == null )  {^[CLASS] IdentityPredicate  [METHOD] identityPredicate [RETURN_TYPE] <T>   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return new IdentityPredicate<T> ( iValue ) ;^50^^^^^46^51^return new IdentityPredicate<T> ( object ) ;^[CLASS] IdentityPredicate  [METHOD] identityPredicate [RETURN_TYPE] <T>   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Argument_Swapping]^return object == iValue;^72^^^^^71^73^return iValue == object;^[CLASS] IdentityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Wrong_Operator]^return iValue != object;^72^^^^^71^73^return iValue == object;^[CLASS] IdentityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^return object;^82^^^^^81^83^return iValue;^[CLASS] IdentityPredicate  [METHOD] getValue [RETURN_TYPE] T   [VARIABLES] T  iValue  object  long  serialVersionUID  boolean  
