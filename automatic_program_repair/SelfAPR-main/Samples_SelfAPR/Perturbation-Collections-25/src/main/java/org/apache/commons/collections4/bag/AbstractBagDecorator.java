[P1_Replace_Type]^private static final  int  serialVersionUID = -3768146017343785417L;^36^^^^^31^41^private static final long serialVersionUID = -3768146017343785417L;^[CLASS] AbstractBagDecorator   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = -3768146017343785417;^36^^^^^31^41^private static final long serialVersionUID = -3768146017343785417L;^[CLASS] AbstractBagDecorator   [VARIABLES] 
[P14_Delete_Statement]^^43^^^^^42^44^super (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] <init> [RETURN_TYPE] AbstractBagDecorator()   [VARIABLES] long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^super ( bag ) ;super (  ) ;^43^^^^^42^44^super (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] <init> [RETURN_TYPE] AbstractBagDecorator()   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^53^^^^^52^54^super ( bag ) ;^[CLASS] AbstractBagDecorator  [METHOD] <init> [RETURN_TYPE] Bag)   Bag<E> bag [VARIABLES] Bag  bag  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^super (  ) ;super ( bag ) ;^53^^^^^52^54^super ( bag ) ;^[CLASS] AbstractBagDecorator  [METHOD] <init> [RETURN_TYPE] Bag)   Bag<E> bag [VARIABLES] Bag  bag  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^63^^^^^62^64^return  ( Bag<E> )  super.decorated (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] decorated [RETURN_TYPE] Bag   [VARIABLES] long  serialVersionUID  boolean  
[P2_Replace_Operator]^return object == this && decorated (  ) .equals ( object ) ;^68^^^^^67^69^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P2_Replace_Operator]^return object >= this || decorated (  ) .equals ( object ) ;^68^^^^^67^69^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P7_Replace_Invocation]^return object == this || decorated (  ) .getCount ( object ) ;^68^^^^^67^69^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P14_Delete_Statement]^^68^^^^^67^69^return object == this || decorated (  ) .equals ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P7_Replace_Invocation]^return decorated (  ) .uniqueSet (  ) ;^73^^^^^72^74^return decorated (  ) .hashCode (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^73^^^^^72^74^return decorated (  ) .hashCode (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] long  serialVersionUID  boolean  
[P7_Replace_Invocation]^return decorated (  ) .equals ( object ) ;^79^^^^^78^80^return decorated (  ) .getCount ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] getCount [RETURN_TYPE] int   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P14_Delete_Statement]^^79^^^^^78^80^return decorated (  ) .getCount ( object ) ;^[CLASS] AbstractBagDecorator  [METHOD] getCount [RETURN_TYPE] int   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P5_Replace_Variable]^return decorated (  ) .add (  count ) ;^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P5_Replace_Variable]^return decorated (  ) .add ( object ) ;^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P5_Replace_Variable]^return decorated (  ) .add ( count, object ) ;^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P7_Replace_Invocation]^return decorated (  ) .remove ( object, count ) ;^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P7_Replace_Invocation]^return decorated (  )  .hashCode (  )  ;^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P14_Delete_Statement]^^83^^^^^82^84^return decorated (  ) .add ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[P5_Replace_Variable]^return decorated (  ) .remove (  count ) ;^87^^^^^86^88^return decorated (  ) .remove ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[P5_Replace_Variable]^return decorated (  ) .remove ( object ) ;^87^^^^^86^88^return decorated (  ) .remove ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[P5_Replace_Variable]^return decorated (  ) .remove ( count, object ) ;^87^^^^^86^88^return decorated (  ) .remove ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[P14_Delete_Statement]^^87^^^^^86^88^return decorated (  ) .remove ( object, count ) ;^[CLASS] AbstractBagDecorator  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[P7_Replace_Invocation]^return decorated (  ) .hashCode (  ) ;^91^^^^^90^92^return decorated (  ) .uniqueSet (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] uniqueSet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
[P14_Delete_Statement]^^91^^^^^90^92^return decorated (  ) .uniqueSet (  ) ;^[CLASS] AbstractBagDecorator  [METHOD] uniqueSet [RETURN_TYPE] Set   [VARIABLES] long  serialVersionUID  boolean  
