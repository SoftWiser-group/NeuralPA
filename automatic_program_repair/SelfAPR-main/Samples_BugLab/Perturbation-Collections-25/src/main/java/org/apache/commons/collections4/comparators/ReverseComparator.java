[BugLab_Wrong_Operator]^this.comparator = comparator != null ? ComparatorUtils.NATURAL_COMPARATOR : comparator;^64^^^^^63^65^this.comparator = comparator == null ? ComparatorUtils.NATURAL_COMPARATOR : comparator;^[CLASS] ReverseComparator  [METHOD] <init> [RETURN_TYPE] Comparator)   Comparator<? super E> comparator [VARIABLES] long  serialVersionUID  Comparator  comparator  boolean  
[BugLab_Variable_Misuse]^return comparator.compare ( obj2, obj2 ) ;^76^^^^^75^77^return comparator.compare ( obj2, obj1 ) ;^[CLASS] ReverseComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] boolean  E  obj1  obj2  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return comparator.compare ( obj1, obj2 ) ;^76^^^^^75^77^return comparator.compare ( obj2, obj1 ) ;^[CLASS] ReverseComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] boolean  E  obj1  obj2  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return obj1.compare ( obj2, comparator ) ;^76^^^^^75^77^return comparator.compare ( obj2, obj1 ) ;^[CLASS] ReverseComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] boolean  E  obj1  obj2  long  serialVersionUID  Comparator  comparator  
[BugLab_Variable_Misuse]^return comparator.compare ( obj1, obj1 ) ;^76^^^^^75^77^return comparator.compare ( obj2, obj1 ) ;^[CLASS] ReverseComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] boolean  E  obj1  obj2  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return obj2.compare ( comparator, obj1 ) ;^76^^^^^75^77^return comparator.compare ( obj2, obj1 ) ;^[CLASS] ReverseComparator  [METHOD] compare [RETURN_TYPE] int   final E obj1 final E obj2 [VARIABLES] boolean  E  obj1  obj2  long  serialVersionUID  Comparator  comparator  
[BugLab_Wrong_Operator]^if  ( this >= object )  {^110^^^^^109^121^if  ( this == object )  {^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Wrong_Literal]^return false;^111^^^^^109^121^return true;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Wrong_Operator]^if  ( null != object )  {^113^^^^^109^121^if  ( null == object )  {^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Wrong_Literal]^return true;^114^^^^^109^121^return false;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return comparator.equals ( thatrc.comparator.comparator ) ;^118^^^^^109^121^return comparator.equals ( thatrc.comparator ) ;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return thatrc.equals ( comparator.comparator ) ;^118^^^^^109^121^return comparator.equals ( thatrc.comparator ) ;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return thatrc.comparator.equals ( comparator ) ;^118^^^^^109^121^return comparator.equals ( thatrc.comparator ) ;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Argument_Swapping]^return comparator.equals ( thatrc ) ;^118^^^^^109^121^return comparator.equals ( thatrc.comparator ) ;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
[BugLab_Wrong_Literal]^return true;^120^^^^^109^121^return false;^[CLASS] ReverseComparator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  ReverseComparator  thatrc  boolean  long  serialVersionUID  Comparator  comparator  
