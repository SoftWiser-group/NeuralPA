[BugLab_Wrong_Operator]^if  ( set  <  Unmodifiable )  {^55^^^^^54^59^if  ( set instanceof Unmodifiable )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] unmodifiableEntrySet [RETURN_TYPE] <K,V>   Entry<K, V>> set [VARIABLES] long  serialVersionUID  Set  set  boolean  
[BugLab_Variable_Misuse]^return this;^56^^^^^54^59^return set;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] unmodifiableEntrySet [RETURN_TYPE] <K,V>   Entry<K, V>> set [VARIABLES] long  serialVersionUID  Set  set  boolean  
[BugLab_Variable_Misuse]^return 3;^56^^^^^54^59^return set;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] unmodifiableEntrySet [RETURN_TYPE] <K,V>   Entry<K, V>> set [VARIABLES] long  serialVersionUID  Set  set  boolean  
[BugLab_Argument_Swapping]^for  ( arraynt i = 0; i < i.length; i++ )  {^113^^^^^111^117^for  ( int i = 0; i < array.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] Object[]   [VARIABLES] boolean  long  serialVersionUID  Object[]  array  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < array.length.length; i++ )  {^113^^^^^111^117^for  ( int i = 0; i < array.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] Object[]   [VARIABLES] boolean  long  serialVersionUID  Object[]  array  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < array; i++ )  {^113^^^^^111^117^for  ( int i = 0; i < array.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] Object[]   [VARIABLES] boolean  long  serialVersionUID  Object[]  array  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= array.length; i++ )  {^113^^^^^111^117^for  ( int i = 0; i < array.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] Object[]   [VARIABLES] boolean  long  serialVersionUID  Object[]  array  int  i  
[BugLab_Wrong_Literal]^for  ( int i = ; i < array.length; i++ )  {^113^^^^^111^117^for  ( int i = 0; i < array.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] Object[]   [VARIABLES] boolean  long  serialVersionUID  Object[]  array  int  i  
[BugLab_Variable_Misuse]^if  ( i > 0 )  {^123^^^^^121^144^if  ( array.length > 0 )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( array.length.length > 0 )  {^123^^^^^121^144^if  ( array.length > 0 )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( array > 0 )  {^123^^^^^121^144^if  ( array.length > 0 )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Operator]^if  ( array.length >= 0 )  {^123^^^^^121^144^if  ( array.length > 0 )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Literal]^if  ( array.length > i )  {^123^^^^^121^144^if  ( array.length > 0 )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Literal]^result =  ( Object[] )  Array.newInstance ( array.getClass (  ) .getComponentType (  ) , 1 ) ;^126^^^^^121^144^result =  ( Object[] )  Array.newInstance ( array.getClass (  ) .getComponentType (  ) , 0 ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Literal]^result =  ( Object[] )  Array.newInstance ( array.getClass (  ) .getComponentType (  ) , i ) ;^126^^^^^121^144^result =  ( Object[] )  Array.newInstance ( array.getClass (  ) .getComponentType (  ) , 0 ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^for  ( result.lengthnt i = 0; i < i; i++ )  {^129^^^^^121^144^for  ( int i = 0; i < result.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < result.length.length; i++ )  {^129^^^^^121^144^for  ( int i = 0; i < result.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= result.length; i++ )  {^129^^^^^121^144^for  ( int i = 0; i < result.length; i++ )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Variable_Misuse]^if  ( i > array.length )  {^134^^^^^121^144^if  ( result.length > array.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Variable_Misuse]^if  ( result.length > i )  {^134^^^^^121^144^if  ( result.length > array.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( array.length.length > result )  {^134^^^^^121^144^if  ( result.length > array.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( array.length > result.length )  {^134^^^^^121^144^if  ( result.length > array.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Operator]^if  ( result.length >= array.length )  {^134^^^^^121^144^if  ( result.length > array.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Variable_Misuse]^System.arraycopy ( result, 0, array, 0, i ) ;^139^^^^^121^144^System.arraycopy ( result, 0, array, 0, result.length ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^System.arraycopy ( result.length, 0, array, 0, result ) ;^139^^^^^121^144^System.arraycopy ( result, 0, array, 0, result.length ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^System.arraycopy ( result, 0, result.length, 0, array ) ;^139^^^^^121^144^System.arraycopy ( result, 0, array, 0, result.length ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Literal]^System.arraycopy ( result, i, array, i, result.length ) ;^139^^^^^121^144^System.arraycopy ( result, 0, array, 0, result.length ) ;^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Variable_Misuse]^if  ( i > result.length )  {^140^^^^^121^144^if  ( array.length > result.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Variable_Misuse]^if  ( array.length > i )  {^140^^^^^121^144^if  ( array.length > result.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( result.length > array.length )  {^140^^^^^121^144^if  ( array.length > result.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Argument_Swapping]^if  ( array.length > result.length.length )  {^140^^^^^121^144^if  ( array.length > result.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  
[BugLab_Wrong_Operator]^if  ( array.length < result.length )  {^140^^^^^121^144^if  ( array.length > result.length )  {^[CLASS] UnmodifiableEntrySet UnmodifiableEntrySetIterator UnmodifiableEntry  [METHOD] toArray [RETURN_TYPE] <T>   final T[] array [VARIABLES] boolean  long  serialVersionUID  Object[]  result  int  i  T[]  array  