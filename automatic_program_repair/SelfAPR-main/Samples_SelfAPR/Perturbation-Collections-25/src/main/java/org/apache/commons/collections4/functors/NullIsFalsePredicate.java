[P1_Replace_Type]^private static final  int  serialVersionUID = -2997501534564735525L;^32^^^^^27^37^private static final long serialVersionUID = -2997501534564735525L;^[CLASS] NullIsFalsePredicate   [VARIABLES] 
[P8_Replace_Mix]^private static  long serialVersionUID = -2997501534564735525;^32^^^^^27^37^private static final long serialVersionUID = -2997501534564735525L;^[CLASS] NullIsFalsePredicate   [VARIABLES] 
[P14_Delete_Statement]^^59^60^^^^58^61^super (  ) ; iPredicate = predicate;^[CLASS] NullIsFalsePredicate  [METHOD] <init> [RETURN_TYPE] Predicate)   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P8_Replace_Mix]^iPredicate =  null;^60^^^^^58^61^iPredicate = predicate;^[CLASS] NullIsFalsePredicate  [METHOD] <init> [RETURN_TYPE] Predicate)   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P2_Replace_Operator]^if  ( predicate != null )  {^46^^^^^45^50^if  ( predicate == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P8_Replace_Mix]^if  ( predicate == false )  {^46^^^^^45^50^if  ( predicate == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P9_Replace_Statement]^if  ( object == null )  {^46^^^^^45^50^if  ( predicate == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException("Predicate must not be null");^46^47^48^^^45^50^if  ( predicate == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P16_Remove_Block]^^46^47^48^^^45^50^if  ( predicate == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P13_Insert_Block]^if  ( predicate == null )  {     throw new IllegalArgumentException ( "Predicate must not be null" ) ; }^47^^^^^45^50^[Delete]^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P8_Replace_Mix]^return ;^47^^^^^45^50^throw new IllegalArgumentException  (" ")  ;^[CLASS] NullIsFalsePredicate  [METHOD] nullIsFalsePredicate [RETURN_TYPE] <T>   Predicate<? super T> predicate [VARIABLES] long  serialVersionUID  Predicate  iPredicate  predicate  boolean  
[P2_Replace_Operator]^if  ( object != null )  {^71^^^^^70^75^if  ( object == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P8_Replace_Mix]^if  ( object == false )  {^71^^^^^70^75^if  ( object == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P9_Replace_Statement]^if  ( predicate == null )  {^71^^^^^70^75^if  ( object == null )  {^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P15_Unwrap_Block]^return false;^71^72^73^^^70^75^if  ( object == null )  { return false; }^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P16_Remove_Block]^^71^72^73^^^70^75^if  ( object == null )  { return false; }^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P3_Replace_Literal]^return true;^72^^^^^70^75^return false;^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P5_Replace_Variable]^return object.evaluate ( iPredicate ) ;^74^^^^^70^75^return iPredicate.evaluate ( object ) ;^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  
[P14_Delete_Statement]^^74^^^^^70^75^return iPredicate.evaluate ( object ) ;^[CLASS] NullIsFalsePredicate  [METHOD] evaluate [RETURN_TYPE] boolean   final T object [VARIABLES] boolean  T  object  long  serialVersionUID  Predicate  iPredicate  predicate  