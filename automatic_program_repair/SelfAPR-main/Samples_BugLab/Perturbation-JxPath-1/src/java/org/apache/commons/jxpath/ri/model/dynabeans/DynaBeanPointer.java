[BugLab_Wrong_Literal]^return true;^73^^^^^72^74^return false;^[CLASS] DynaBeanPointer  [METHOD] isCollection [RETURN_TYPE] boolean   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
[BugLab_Wrong_Literal]^return 2;^80^^^^^79^81^return 1;^[CLASS] DynaBeanPointer  [METHOD] getLength [RETURN_TYPE] int   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
[BugLab_Wrong_Literal]^return true;^84^^^^^83^85^return false;^[CLASS] DynaBeanPointer  [METHOD] isLeaf [RETURN_TYPE] boolean   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
[BugLab_Wrong_Operator]^return name != null ? 0 : name.hashCode (  ) ;^88^^^^^87^89^return name == null ? 0 : name.hashCode (  ) ;^[CLASS] DynaBeanPointer  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
[BugLab_Wrong_Literal]^return name == null ? 1 : name.hashCode (  ) ;^88^^^^^87^89^return name == null ? 0 : name.hashCode (  ) ;^[CLASS] DynaBeanPointer  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
[BugLab_Wrong_Operator]^if  ( object <= this )  {^92^^^^^91^119^if  ( object == this )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return false;^93^^^^^91^119^return true;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( ! ( object  >>  DynaBeanPointer )  )  {^96^^^^^91^119^if  ( ! ( object instanceof DynaBeanPointer )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return true;^97^^^^^91^119^return false;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( parent <= other.parent )  {^101^^^^^91^119^if  ( parent != other.parent )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( parent == null && !parent.equals ( other.parent )  )  {^102^^^^^91^119^if  ( parent == null || !parent.equals ( other.parent )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( parent != null || !parent.equals ( other.parent )  )  {^102^^^^^91^119^if  ( parent == null || !parent.equals ( other.parent )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return true;^103^^^^^91^119^return false;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^if  (  ( other == null && name.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^if  (  ( other.name == null && name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^if  (  ( name == null && other != null ) ||  ( name != null && !name.equals ( other.name.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  (  ( name == null && other.name != null ) &&  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  (  ( name == null || other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  (  ( name != null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  (  ( name == null && other.name == null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  (  ( name == null && other.name == null ) ||  ( name == null && !name.equals ( other.name )  )  )  {^107^108^^^^91^119^if  (  ( name == null && other.name != null ) ||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return true;^109^^^^^107^110^return false;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return true;^109^^^^^91^119^return false;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^||  ( other != null && !name.equals ( name.name )  )  )  {^108^^^^^91^119^||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^||  ( other.name != null && !name.equals ( name )  )  )  {^108^^^^^91^119^||  ( name != null && !name.equals ( other.name )  )  )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Variable_Misuse]^int iThis =  ( other == WHOLE_COLLECTION ? 0 : index ) ;^112^^^^^91^119^int iThis =  ( index == WHOLE_COLLECTION ? 0 : index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Variable_Misuse]^int iThis =  ( index == other ? 0 : index ) ;^112^^^^^91^119^int iThis =  ( index == WHOLE_COLLECTION ? 0 : index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^int iThis =  ( WHOLE_COLLECTION == index ? 0 : index ) ;^112^^^^^91^119^int iThis =  ( index == WHOLE_COLLECTION ? 0 : index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^int iThis =  ( index > WHOLE_COLLECTION ? 0 : index ) ;^112^^^^^91^119^int iThis =  ( index == WHOLE_COLLECTION ? 0 : index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^int iThis =  ( index == WHOLE_COLLECTION ? iThis : index ) ;^112^^^^^91^119^int iThis =  ( index == WHOLE_COLLECTION ? 0 : index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^int iOther =  ( other.index != WHOLE_COLLECTION ? 0 : other.index ) ;^113^^^^^91^119^int iOther =  ( other.index == WHOLE_COLLECTION ? 0 : other.index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^int iOther =  ( other.index == WHOLE_COLLECTION ? -1 : other.index ) ;^113^^^^^91^119^int iOther =  ( other.index == WHOLE_COLLECTION ? 0 : other.index ) ;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^if  ( iOther != iThis )  {^114^^^^^91^119^if  ( iThis != iOther )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( iThis <= iOther )  {^114^^^^^91^119^if  ( iThis != iOther )  {^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Literal]^return true;^115^^^^^91^119^return false;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Variable_Misuse]^return dynaBean == dynaBean;^118^^^^^91^119^return dynaBean == other.dynaBean;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^return dynaBean == other.dynaBean.dynaBean;^118^^^^^91^119^return dynaBean == other.dynaBean;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^return other == dynaBean.dynaBean;^118^^^^^91^119^return dynaBean == other.dynaBean;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Argument_Swapping]^return other.dynaBean == dynaBean;^118^^^^^91^119^return dynaBean == other.dynaBean;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^return dynaBean < other.dynaBean;^118^^^^^91^119^return dynaBean == other.dynaBean;^[CLASS] DynaBeanPointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  DynaBeanPointer  other  boolean  DynaBean  dynaBean  QName  name  int  iOther  iThis  
[BugLab_Wrong_Operator]^if  ( parent == null )  {^125^^^^^124^129^if  ( parent != null )  {^[CLASS] DynaBeanPointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] DynaBean  dynaBean  QName  name  boolean  
