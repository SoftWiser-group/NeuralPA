[P1_Replace_Type]^private static final  short  serialVersionUID = -684521469108685117L;^43^^^^^38^48^private static final long serialVersionUID = -684521469108685117L;^[CLASS] PredicatedSet   [VARIABLES] 
[P8_Replace_Mix]^private static final  int  serialVersionUID = -684521469108685117;^43^^^^^38^48^private static final long serialVersionUID = -684521469108685117L;^[CLASS] PredicatedSet   [VARIABLES] 
[P5_Replace_Variable]^super (  predicate ) ;^76^^^^^75^77^super ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] <init> [RETURN_TYPE] Predicate)   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P5_Replace_Variable]^super ( set ) ;^76^^^^^75^77^super ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] <init> [RETURN_TYPE] Predicate)   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P5_Replace_Variable]^super ( predicate, set ) ;^76^^^^^75^77^super ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] <init> [RETURN_TYPE] Predicate)   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P14_Delete_Statement]^^76^^^^^75^77^super ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] <init> [RETURN_TYPE] Predicate)   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P4_Replace_Constructor]^return new PredicatedSet<E> (  predicate ) ;^60^^^^^59^61^return new PredicatedSet<E> ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] predicatedSet [RETURN_TYPE] <E>   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P4_Replace_Constructor]^return new PredicatedSet<E> ( set ) ;^60^^^^^59^61^return new PredicatedSet<E> ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] predicatedSet [RETURN_TYPE] <E>   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P5_Replace_Variable]^return new PredicatedSet<E> ( predicate, set ) ;^60^^^^^59^61^return new PredicatedSet<E> ( set, predicate ) ;^[CLASS] PredicatedSet  [METHOD] predicatedSet [RETURN_TYPE] <E>   Set<E> set Predicate<? super E> predicate [VARIABLES] Set  set  boolean  long  serialVersionUID  Predicate  predicate  
[P8_Replace_Mix]^return  ( Set<E> )  super .decorated (  )  ;^86^^^^^85^87^return  ( Set<E> )  super.decorated (  ) ;^[CLASS] PredicatedSet  [METHOD] decorated [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^86^^^^^85^87^return  ( Set<E> )  super.decorated (  ) ;^[CLASS] PredicatedSet  [METHOD] decorated [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P2_Replace_Operator]^return object == this && decorated (  ) .equals ( object ) ;^91^^^^^90^92^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] PredicatedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P2_Replace_Operator]^return object >= this || decorated (  ) .equals ( object ) ;^91^^^^^90^92^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] PredicatedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P8_Replace_Mix]^return object ;^91^^^^^90^92^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] PredicatedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P14_Delete_Statement]^^91^^^^^90^92^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] PredicatedSet  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P7_Replace_Invocation]^return decorated (  )  .hashCode (  )  ;^96^^^^^95^97^return decorated (  ) .hashCode (  ) ;^[CLASS] PredicatedSet  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^96^^^^^95^97^return decorated (  ) .hashCode (  ) ;^[CLASS] PredicatedSet  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  