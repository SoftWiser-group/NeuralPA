[BugLab_Wrong_Operator]^if  ( prefix != null )  {^49^^^^^47^53^if  ( prefix == null )  {^[CLASS] DOMAttributePointer  [METHOD] getNamespaceURI [RETURN_TYPE] String   [VARIABLES] Attr  attr  String  prefix  boolean  
[BugLab_Argument_Swapping]^return prefix.getNamespaceURI ( parent ) ;^52^^^^^47^53^return parent.getNamespaceURI ( prefix ) ;^[CLASS] DOMAttributePointer  [METHOD] getNamespaceURI [RETURN_TYPE] String   [VARIABLES] Attr  attr  String  prefix  boolean  
[BugLab_Wrong_Operator]^if  ( value != null )  {^57^^^^^55^64^if  ( value == null )  {^[CLASS] DOMAttributePointer  [METHOD] getValue [RETURN_TYPE] Object   [VARIABLES] Attr  attr  String  value  boolean  
[BugLab_Wrong_Operator]^if  ( value.equals ( "" )  || !attr.getSpecified (  )  )  {^60^^^^^55^64^if  ( value.equals ( "" )  && !attr.getSpecified (  )  )  {^[CLASS] DOMAttributePointer  [METHOD] getValue [RETURN_TYPE] Object   [VARIABLES] Attr  attr  String  value  boolean  
[BugLab_Wrong_Literal]^return true;^71^^^^^70^72^return false;^[CLASS] DOMAttributePointer  [METHOD] isCollection [RETURN_TYPE] boolean   [VARIABLES] Attr  attr  boolean  
[BugLab_Wrong_Literal]^return false;^83^^^^^82^84^return true;^[CLASS] DOMAttributePointer  [METHOD] isActual [RETURN_TYPE] boolean   [VARIABLES] Attr  attr  boolean  
[BugLab_Wrong_Literal]^return false;^87^^^^^86^88^return true;^[CLASS] DOMAttributePointer  [METHOD] isLeaf [RETURN_TYPE] boolean   [VARIABLES] Attr  attr  boolean  
[BugLab_Wrong_Operator]^if  ( parent == null )  {^112^^^^^110^122^if  ( parent != null )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  == 0 && buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  != 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  <= '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )   &&  1 )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Literal]^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )   )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^|| buffer.charAt ( buffer.length (  )   &  1 )  != '/' )  {^115^^^^^110^122^|| buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Literal]^|| buffer.charAt ( buffer.length (  )   )  != '/' )  {^115^^^^^110^122^|| buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  < 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  > '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )   |  1 )  != '/' )  {^114^115^^^^110^122^if  ( buffer.length (  )  == 0 || buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Variable_Misuse]^buffer.append ( this.asPath (  )  ) ;^113^^^^^110^122^buffer.append ( parent.asPath (  )  ) ;^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^|| buffer.charAt ( buffer.length (  )   <  1 )  != '/' )  {^115^^^^^110^122^|| buffer.charAt ( buffer.length (  )  - 1 )  != '/' )  {^[CLASS] DOMAttributePointer  [METHOD] asPath [RETURN_TYPE] String   [VARIABLES] Attr  attr  StringBuffer  buffer  boolean  
[BugLab_Wrong_Operator]^if  ( object <= this )  {^129^^^^^128^139^if  ( object == this )  {^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Wrong_Literal]^return false;^130^^^^^128^139^return true;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Wrong_Operator]^if  ( ! ( object  ||  DOMAttributePointer )  )  {^133^^^^^128^139^if  ( ! ( object instanceof DOMAttributePointer )  )  {^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Wrong_Literal]^return true;^134^^^^^128^139^return false;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Variable_Misuse]^return attr == attr;^138^^^^^128^139^return attr == other.attr;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Argument_Swapping]^return attr == other.attr.attr;^138^^^^^128^139^return attr == other.attr;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Argument_Swapping]^return other.attr == attr;^138^^^^^128^139^return attr == other.attr;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Argument_Swapping]^return attr == other;^138^^^^^128^139^return attr == other.attr;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Wrong_Operator]^return attr >= other.attr;^138^^^^^128^139^return attr == other.attr;^[CLASS] DOMAttributePointer  [METHOD] equals [RETURN_TYPE] boolean   Object object [VARIABLES] Object  object  boolean  DOMAttributePointer  other  Attr  attr  
[BugLab_Wrong_Literal]^return ;^146^^^^^141^147^return 0;^[CLASS] DOMAttributePointer  [METHOD] compareChildNodePointers [RETURN_TYPE] int   NodePointer pointer1 NodePointer pointer2 [VARIABLES] Attr  attr  NodePointer  pointer1  pointer2  boolean  
