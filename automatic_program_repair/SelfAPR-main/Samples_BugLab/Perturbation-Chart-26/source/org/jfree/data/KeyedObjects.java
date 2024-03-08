[BugLab_Variable_Misuse]^return data.size (  ) ;^77^^^^^76^78^return this.data.size (  ) ;^[CLASS] KeyedObjects  [METHOD] getItemCount [RETURN_TYPE] int   [VARIABLES] List  data  long  serialVersionUID  boolean  
[BugLab_Variable_Misuse]^if  ( item >= 0 && item < data.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Argument_Swapping]^if  ( this.data >= 0 && item < item.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Wrong_Operator]^if  ( item >= 0 || item < this.data.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Wrong_Operator]^if  ( item > 0 && item < this.data.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Wrong_Operator]^if  ( item >= 0 && item <= this.data.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Wrong_Literal]^if  ( item >=  && item < this.data.size (  )  )  {^89^^^^^87^96^if  ( item >= 0 && item < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Wrong_Operator]^if  ( kobj == null )  {^91^^^^^87^96^if  ( kobj != null )  {^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Variable_Misuse]^KeyedObject kobj =  ( KeyedObject )  data.get ( item ) ;^90^^^^^87^96^KeyedObject kobj =  ( KeyedObject )  this.data.get ( item ) ;^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Argument_Swapping]^KeyedObject kobj =  ( KeyedObject )  item.get ( this.data ) ;^90^^^^^87^96^KeyedObject kobj =  ( KeyedObject )  this.data.get ( item ) ;^[CLASS] KeyedObjects  [METHOD] getObject [RETURN_TYPE] Object   int item [VARIABLES] KeyedObject  kobj  List  data  Object  result  boolean  long  serialVersionUID  int  item  
[BugLab_Variable_Misuse]^if  ( index >= 0 && index < data.size (  )  )  {^109^^^^^107^116^if  ( index >= 0 && index < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Argument_Swapping]^if  ( this.data >= 0 && index < index.size (  )  )  {^109^^^^^107^116^if  ( index >= 0 && index < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Wrong_Operator]^if  ( index >= 0 || index < this.data.size (  )  )  {^109^^^^^107^116^if  ( index >= 0 && index < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Wrong_Operator]^if  ( index > 0 && index < this.data.size (  )  )  {^109^^^^^107^116^if  ( index >= 0 && index < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Wrong_Operator]^if  ( index >= 0 && index <= this.data.size (  )  )  {^109^^^^^107^116^if  ( index >= 0 && index < this.data.size (  )  )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Wrong_Operator]^if  ( item == null )  {^111^^^^^107^116^if  ( item != null )  {^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Variable_Misuse]^KeyedObject item =  ( KeyedObject )  data.get ( index ) ;^110^^^^^107^116^KeyedObject item =  ( KeyedObject )  this.data.get ( index ) ;^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Argument_Swapping]^KeyedObject item =  ( KeyedObject )  index.get ( this.data ) ;^110^^^^^107^116^KeyedObject item =  ( KeyedObject )  this.data.get ( index ) ;^[CLASS] KeyedObjects  [METHOD] getKey [RETURN_TYPE] Comparable   int index [VARIABLES] KeyedObject  item  List  data  Comparable  result  boolean  long  serialVersionUID  int  index  
[BugLab_Wrong_Literal]^int result = -result;^126^^^^^125^137^int result = -1;^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Wrong_Literal]^int i = result;^127^^^^^125^137^int i = 0;^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Variable_Misuse]^Iterator iterator = data.iterator (  ) ;^128^^^^^125^137^Iterator iterator = this.data.iterator (  ) ;^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Argument_Swapping]^while  ( i.hasNext (  )  )  {^129^^^^^125^137^while  ( iterator.hasNext (  )  )  {^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Argument_Swapping]^if  ( key.getKey (  ) .equals ( ko )  )  {^131^^^^^125^137^if  ( ko.getKey (  ) .equals ( key )  )  {^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Variable_Misuse]^result = result;^132^^^^^125^137^result = i;^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Variable_Misuse]^return i;^136^^^^^125^137^return result;^[CLASS] KeyedObjects  [METHOD] getIndex [RETURN_TYPE] int   Comparable key [VARIABLES] KeyedObject  ko  Comparable  key  boolean  Iterator  iterator  List  data  long  serialVersionUID  int  i  result  
[BugLab_Variable_Misuse]^Iterator iterator = result.iterator (  ) ;^146^^^^^144^152^Iterator iterator = this.data.iterator (  ) ;^[CLASS] KeyedObjects  [METHOD] getKeys [RETURN_TYPE] List   [VARIABLES] Iterator  iterator  KeyedObject  ko  List  data  result  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return data;^151^^^^^144^152^return result;^[CLASS] KeyedObjects  [METHOD] getKeys [RETURN_TYPE] List   [VARIABLES] Iterator  iterator  KeyedObject  ko  List  data  result  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^setObject ( object, key ) ;^174^^^^^173^175^setObject ( key, object ) ;^[CLASS] KeyedObjects  [METHOD] addObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] List  data  result  Comparable  key  Object  object  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( keyIndex < 0 )  {^187^^^^^185^195^if  ( keyIndex >= 0 )  {^[CLASS] KeyedObjects  [METHOD] setObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] KeyedObject  ko  Comparable  key  boolean  List  data  result  Object  object  long  serialVersionUID  int  keyIndex  
[BugLab_Wrong_Literal]^if  ( keyIndex >= keyIndex )  {^187^^^^^185^195^if  ( keyIndex >= 0 )  {^[CLASS] KeyedObjects  [METHOD] setObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] KeyedObject  ko  Comparable  key  boolean  List  data  result  Object  object  long  serialVersionUID  int  keyIndex  
[BugLab_Argument_Swapping]^KeyedObject ko = new KeyedObject ( object, key ) ;^192^^^^^185^195^KeyedObject ko = new KeyedObject ( key, object ) ;^[CLASS] KeyedObjects  [METHOD] setObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] KeyedObject  ko  Comparable  key  boolean  List  data  result  Object  object  long  serialVersionUID  int  keyIndex  
[BugLab_Variable_Misuse]^KeyedObject ko =  ( KeyedObject )  result.get ( keyIndex ) ;^188^^^^^185^195^KeyedObject ko =  ( KeyedObject )  this.data.get ( keyIndex ) ;^[CLASS] KeyedObjects  [METHOD] setObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] KeyedObject  ko  Comparable  key  boolean  List  data  result  Object  object  long  serialVersionUID  int  keyIndex  
[BugLab_Argument_Swapping]^KeyedObject ko =  ( KeyedObject )  keyIndex.get ( this.data ) ;^188^^^^^185^195^KeyedObject ko =  ( KeyedObject )  this.data.get ( keyIndex ) ;^[CLASS] KeyedObjects  [METHOD] setObject [RETURN_TYPE] void   Comparable key Object object [VARIABLES] KeyedObject  ko  Comparable  key  boolean  List  data  result  Object  object  long  serialVersionUID  int  keyIndex  
[BugLab_Variable_Misuse]^Iterator iterator = result.iterator (  ) ;^225^^^^^222^231^Iterator iterator = this.data.iterator (  ) ;^[CLASS] KeyedObjects  [METHOD] clone [RETURN_TYPE] Object   [VARIABLES] KeyedObjects  clone  Iterator  iterator  KeyedObject  ko  List  data  result  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( o2 == null )  {^242^^^^^227^257^if  ( o == null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( o != null )  {^242^^^^^227^257^if  ( o == null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^243^^^^^228^258^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( o >= this )  {^245^^^^^230^260^if  ( o == this )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return false;^246^^^^^231^261^return true;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( ! ( o2 instanceof KeyedObjects )  )  {^249^^^^^234^264^if  ( ! ( o instanceof KeyedObjects )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( ! ( o  &  KeyedObjects )  )  {^249^^^^^234^264^if  ( ! ( o instanceof KeyedObjects )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^250^^^^^235^265^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( i != kos.getItemCount (  )  )  {^255^^^^^240^270^if  ( count != kos.getItemCount (  )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^if  ( kos != count.getItemCount (  )  )  {^255^^^^^240^270^if  ( count != kos.getItemCount (  )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( count <= kos.getItemCount (  )  )  {^255^^^^^240^270^if  ( count != kos.getItemCount (  )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^256^^^^^241^271^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^263^^^^^248^278^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( o == null )  {^267^^^^^252^282^if  ( o1 == null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( o1 != null )  {^267^^^^^252^282^if  ( o1 == null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( !o1.equals ( o )  )  {^273^^^^^267^276^if  ( !o1.equals ( o2 )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^274^^^^^267^276^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( o1 != null )  {^268^^^^^253^283^if  ( o2 != null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^if  ( o2 == null )  {^268^^^^^253^283^if  ( o2 != null )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^269^^^^^254^284^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^if  ( !o1.equals ( o )  )  {^273^^^^^258^288^if  ( !o1.equals ( o2 )  )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return true;^274^^^^^259^289^return false;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^for  ( countnt i = 0; i < count; i++ )  {^259^^^^^244^274^for  ( int i = 0; i < count; i++ )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^for  ( countnt i = 0; i < i; i++ )  {^259^^^^^244^274^for  ( int i = 0; i < count; i++ )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= count; i++ )  {^259^^^^^244^274^for  ( int i = 0; i < count; i++ )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Comparable k1 = getKey ( count ) ;^260^^^^^245^275^Comparable k1 = getKey ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Comparable k2 = kos.getKey ( count ) ;^261^^^^^246^276^Comparable k2 = kos.getKey ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^Comparable k2 = i.getKey ( kos ) ;^261^^^^^246^276^Comparable k2 = kos.getKey ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Object o1 = getObject ( count ) ;^265^^^^^250^280^Object o1 = getObject ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Variable_Misuse]^Object o2 = kos.getObject ( count ) ;^266^^^^^251^281^Object o2 = kos.getObject ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Argument_Swapping]^Object o2 = i.getObject ( kos ) ;^266^^^^^251^281^Object o2 = kos.getObject ( i ) ;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < count; i++ )  {^259^^^^^244^274^for  ( int i = 0; i < count; i++ )  {^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  
[BugLab_Wrong_Literal]^return false;^278^^^^^263^293^return true;^[CLASS] KeyedObjects  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Comparable  k1  k2  boolean  KeyedObjects  kos  List  data  result  Object  o  o1  o2  long  serialVersionUID  int  count  i  