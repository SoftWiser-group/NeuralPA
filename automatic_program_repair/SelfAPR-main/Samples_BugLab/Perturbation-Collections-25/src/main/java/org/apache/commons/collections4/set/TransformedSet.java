[BugLab_Variable_Misuse]^super ( null, transformer ) ;^101^^^^^100^102^super ( set, transformer ) ;^[CLASS] TransformedSet  [METHOD] <init> [RETURN_TYPE] Transformer)   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^super ( transformer, set ) ;^101^^^^^100^102^super ( set, transformer ) ;^[CLASS] TransformedSet  [METHOD] <init> [RETURN_TYPE] Transformer)   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new TransformedSet<E> ( transformer, set ) ;^58^^^^^56^59^return new TransformedSet<E> ( set, transformer ) ;^[CLASS] TransformedSet  [METHOD] transformingSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^final TransformedSet<E> decorated = new TransformedSet<E> ( set, null ) ;^77^^^^^76^87^final TransformedSet<E> decorated = new TransformedSet<E> ( set, transformer ) ;^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Argument_Swapping]^final TransformedSet<E> decorated = new TransformedSet<E> ( transformer, set ) ;^77^^^^^76^87^final TransformedSet<E> decorated = new TransformedSet<E> ( set, transformer ) ;^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( transformer != null || set != null && set.size (  )  > 0 )  {^78^^^^^76^87^if  ( transformer != null && set != null && set.size (  )  > 0 )  {^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( transformer == null && set != null && set.size (  )  > 0 )  {^78^^^^^76^87^if  ( transformer != null && set != null && set.size (  )  > 0 )  {^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( transformer != null && set == null && set.size (  )  > 0 )  {^78^^^^^76^87^if  ( transformer != null && set != null && set.size (  )  > 0 )  {^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( transformer != null && set != null && set.size (  )  >= 0 )  {^78^^^^^76^87^if  ( transformer != null && set != null && set.size (  )  > 0 )  {^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Argument_Swapping]^decorated.decorated (  ) .add ( value.transform ( transformer )  ) ;^83^^^^^76^87^decorated.decorated (  ) .add ( transformer.transform ( value )  ) ;^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Variable_Misuse]^decorated.3 (  ) .add ( transformer.transform ( value )  ) ;^83^^^^^76^87^decorated.decorated (  ) .add ( transformer.transform ( value )  ) ;^[CLASS] TransformedSet  [METHOD] transformedSet [RETURN_TYPE] <E>   Set<E> set Transformer<? super E, ? extends E> transformer [VARIABLES] Set  set  Transformer  transformer  boolean  E  value  E[]  values  TransformedSet  decorated  long  serialVersionUID  
[BugLab_Wrong_Operator]^return object == this && decorated (  ) .equals ( object ) ;^106^^^^^105^107^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] TransformedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[BugLab_Wrong_Operator]^return object > this || decorated (  ) .equals ( object ) ;^106^^^^^105^107^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] TransformedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  