[P1_Replace_Type]^private static final  short  serialVersionUID = -725356885467962424L;^44^^^^^39^49^private static final long serialVersionUID = -725356885467962424L;^[CLASS] UnmodifiableSortedSet   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = -725356885467962424;^44^^^^^39^49^private static final long serialVersionUID = -725356885467962424L;^[CLASS] UnmodifiableSortedSet   [VARIABLES] 
[P14_Delete_Statement]^^70^^^^^69^71^super ( set ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] <init> [RETURN_TYPE] SortedSet)   SortedSet<E> set [VARIABLES] SortedSet  set  long  serialVersionUID  boolean  
[P2_Replace_Operator]^if  ( set  >=  Unmodifiable )  {^56^^^^^55^60^if  ( set instanceof Unmodifiable )  {^[CLASS] UnmodifiableSortedSet  [METHOD] unmodifiableSortedSet [RETURN_TYPE] <E>   SortedSet<E> set [VARIABLES] SortedSet  set  long  serialVersionUID  boolean  
[P15_Unwrap_Block]^return set;^56^57^58^^^55^60^if  ( set instanceof Unmodifiable )  { return set; }^[CLASS] UnmodifiableSortedSet  [METHOD] unmodifiableSortedSet [RETURN_TYPE] <E>   SortedSet<E> set [VARIABLES] SortedSet  set  long  serialVersionUID  boolean  
[P16_Remove_Block]^^56^57^58^^^55^60^if  ( set instanceof Unmodifiable )  { return set; }^[CLASS] UnmodifiableSortedSet  [METHOD] unmodifiableSortedSet [RETURN_TYPE] <E>   SortedSet<E> set [VARIABLES] SortedSet  set  long  serialVersionUID  boolean  
[P13_Insert_Block]^if  ( set instanceof Unmodifiable )  {     return set; }^59^^^^^55^60^[Delete]^[CLASS] UnmodifiableSortedSet  [METHOD] unmodifiableSortedSet [RETURN_TYPE] <E>   SortedSet<E> set [VARIABLES] SortedSet  set  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^76^77^^^^75^77^return UnmodifiableIterator.unmodifiableIterator ( decorated (  ) .iterator (  )  ) ; }^[CLASS] UnmodifiableSortedSet  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] long  serialVersionUID  boolean  
[P8_Replace_Mix]^return ;^86^^^^^85^87^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableSortedSet  [METHOD] addAll [RETURN_TYPE] boolean   Collection<? extends E> coll [VARIABLES] Collection  coll  long  serialVersionUID  boolean  
[P8_Replace_Mix]^return ;^96^^^^^95^97^throw new UnsupportedOperationException  (" ")  ;^[CLASS] UnmodifiableSortedSet  [METHOD] remove [RETURN_TYPE] boolean   Object object [VARIABLES] long  serialVersionUID  Object  object  boolean  
[P5_Replace_Variable]^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, fromElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P5_Replace_Variable]^final SortedSet<E> sub = decorated (  ) .subSet (  toElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P5_Replace_Variable]^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P5_Replace_Variable]^final SortedSet<E> sub = decorated (  ) .subSet ( toElement, fromElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P7_Replace_Invocation]^final SortedSet<E> sub = decorated (  )  .headSet ( toElement )  ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P8_Replace_Mix]^final SortedSet<E> sub = decorated (  ) .subSet ( toElement, toElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P14_Delete_Statement]^^112^^^^^111^114^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P7_Replace_Invocation]^return UnmodifiableSortedSet ( sub ) ;^113^^^^^111^114^return unmodifiableSortedSet ( sub ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P14_Delete_Statement]^^113^114^^^^111^114^return unmodifiableSortedSet ( sub ) ; }^[CLASS] UnmodifiableSortedSet  [METHOD] subSet [RETURN_TYPE] SortedSet   final E fromElement final E toElement [VARIABLES] boolean  E  fromElement  toElement  SortedSet  sub  long  serialVersionUID  
[P7_Replace_Invocation]^final SortedSet<E> head = decorated (  ) .tailSet ( toElement ) ;^118^^^^^117^120^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;^118^^^^^117^120^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;^118^^^^^117^120^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P14_Delete_Statement]^^118^119^^^^117^120^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ; return unmodifiableSortedSet ( head ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P7_Replace_Invocation]^return UnmodifiableSortedSet ( head ) ;^119^^^^^117^120^return unmodifiableSortedSet ( head ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P14_Delete_Statement]^^119^120^^^^117^120^return unmodifiableSortedSet ( head ) ; }^[CLASS] UnmodifiableSortedSet  [METHOD] headSet [RETURN_TYPE] SortedSet   final E toElement [VARIABLES] boolean  E  toElement  SortedSet  head  long  serialVersionUID  
[P7_Replace_Invocation]^final SortedSet<E> tail = decorated (  ) .headSet ( fromElement ) ;^124^^^^^123^126^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P8_Replace_Mix]^final SortedSet<E> tail = decorated (  )  .subSet ( fromElement , fromElement )  ;^124^^^^^123^126^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> sub = decorated (  ) .subSet ( fromElement, toElement ) ;final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^124^^^^^123^126^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P11_Insert_Donor_Statement]^final SortedSet<E> head = decorated (  ) .headSet ( toElement ) ;final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^124^^^^^123^126^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P14_Delete_Statement]^^124^125^^^^123^126^final SortedSet<E> tail = decorated (  ) .tailSet ( fromElement ) ; return unmodifiableSortedSet ( tail ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P7_Replace_Invocation]^return UnmodifiableSortedSet ( tail ) ;^125^^^^^123^126^return unmodifiableSortedSet ( tail ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P14_Delete_Statement]^^125^126^^^^123^126^return unmodifiableSortedSet ( tail ) ; }^[CLASS] UnmodifiableSortedSet  [METHOD] tailSet [RETURN_TYPE] SortedSet   final E fromElement [VARIABLES] boolean  E  fromElement  SortedSet  tail  long  serialVersionUID  
[P7_Replace_Invocation]^out.writeObject (  ) ;^136^^^^^135^138^out.defaultWriteObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^136^^^^^135^138^out.defaultWriteObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P11_Insert_Donor_Statement]^in.defaultReadObject (  ) ;out.defaultWriteObject (  ) ;^136^^^^^135^138^out.defaultWriteObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P14_Delete_Statement]^^137^^^^^135^138^out.writeObject ( decorated (  )  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream out [VARIABLES] ObjectOutputStream  out  long  serialVersionUID  boolean  
[P7_Replace_Invocation]^in .readObject ( in )  ;^149^^^^^148^151^in.defaultReadObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P14_Delete_Statement]^^149^^^^^148^151^in.defaultReadObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P11_Insert_Donor_Statement]^out.defaultWriteObject (  ) ;in.defaultReadObject (  ) ;^149^^^^^148^151^in.defaultReadObject (  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P7_Replace_Invocation]^readObject (  ( Collection<E> )  in.readObject (  )  ) ;^150^^^^^148^151^setCollection (  ( Collection<E> )  in.readObject (  )  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
[P14_Delete_Statement]^^150^^^^^148^151^setCollection (  ( Collection<E> )  in.readObject (  )  ) ;^[CLASS] UnmodifiableSortedSet  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream in [VARIABLES] long  serialVersionUID  ObjectInputStream  in  boolean  
