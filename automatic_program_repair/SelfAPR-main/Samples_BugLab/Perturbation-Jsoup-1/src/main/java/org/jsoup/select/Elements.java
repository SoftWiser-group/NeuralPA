[BugLab_Argument_Swapping]^if  ( attributeKey.hasAttr ( element )  ) return element.attr ( attributeKey ) ;^41^42^^^^39^45^if  ( element.hasAttr ( attributeKey )  ) return element.attr ( attributeKey ) ;^[CLASS] Elements  [METHOD] attr [RETURN_TYPE] String   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Argument_Swapping]^return attributeKey.attr ( element ) ;^42^^^^^39^45^return element.attr ( attributeKey ) ;^[CLASS] Elements  [METHOD] attr [RETURN_TYPE] String   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Argument_Swapping]^if  ( attributeKey.hasAttr ( element )  ) return true;^54^55^^^^52^58^if  ( element.hasAttr ( attributeKey )  ) return true;^[CLASS] Elements  [METHOD] hasAttr [RETURN_TYPE] boolean   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Wrong_Literal]^if  ( element.hasAttr ( attributeKey )  ) return false;^54^55^^^^52^58^if  ( element.hasAttr ( attributeKey )  ) return true;^[CLASS] Elements  [METHOD] hasAttr [RETURN_TYPE] boolean   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Wrong_Literal]^return false;^55^^^^^52^58^return true;^[CLASS] Elements  [METHOD] hasAttr [RETURN_TYPE] boolean   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Wrong_Literal]^return true;^57^^^^^52^58^return false;^[CLASS] Elements  [METHOD] hasAttr [RETURN_TYPE] boolean   String attributeKey [VARIABLES] List  contents  elements  String  attributeKey  boolean  Element  element  
[BugLab_Argument_Swapping]^element.attr ( attributeValue, attributeKey ) ;^68^^^^^66^71^element.attr ( attributeKey, attributeValue ) ;^[CLASS] Elements  [METHOD] attr [RETURN_TYPE] Elements   String attributeKey String attributeValue [VARIABLES] List  contents  elements  String  attributeKey  attributeValue  boolean  Element  element  
[BugLab_Argument_Swapping]^if  ( className.hasClass ( element )  ) return true;^128^129^^^^126^132^if  ( element.hasClass ( className )  ) return true;^[CLASS] Elements  [METHOD] hasClass [RETURN_TYPE] boolean   String className [VARIABLES] List  contents  elements  String  className  boolean  Element  element  
[BugLab_Wrong_Literal]^if  ( element.hasClass ( className )  ) return false;^128^129^^^^126^132^if  ( element.hasClass ( className )  ) return true;^[CLASS] Elements  [METHOD] hasClass [RETURN_TYPE] boolean   String className [VARIABLES] List  contents  elements  String  className  boolean  Element  element  
[BugLab_Wrong_Literal]^return false;^129^^^^^126^132^return true;^[CLASS] Elements  [METHOD] hasClass [RETURN_TYPE] boolean   String className [VARIABLES] List  contents  elements  String  className  boolean  Element  element  
[BugLab_Wrong_Literal]^return true;^131^^^^^126^132^return false;^[CLASS] Elements  [METHOD] hasClass [RETURN_TYPE] boolean   String className [VARIABLES] List  contents  elements  String  className  boolean  Element  element  
[BugLab_Wrong_Operator]^if  ( size (  )  < 0 ) return first (  ) .val (  ) ;^140^141^^^^139^144^if  ( size (  )  > 0 ) return first (  ) .val (  ) ;^[CLASS] Elements  [METHOD] val [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Literal]^if  ( size (  )  > -1 ) return first (  ) .val (  ) ;^140^141^^^^139^144^if  ( size (  )  > 0 ) return first (  ) .val (  ) ;^[CLASS] Elements  [METHOD] val [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Operator]^if  ( sb.length (  )  == 0 ) sb.append ( " " ) ;^168^169^^^^165^173^if  ( sb.length (  )  != 0 ) sb.append ( " " ) ;^[CLASS] Elements  [METHOD] text [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  StringBuilder  sb  Element  element  
[BugLab_Wrong_Literal]^if  ( element.hasText (  )  ) return false;^177^178^^^^175^181^if  ( element.hasText (  )  ) return true;^[CLASS] Elements  [METHOD] hasText [RETURN_TYPE] boolean   [VARIABLES] List  contents  elements  Element  element  boolean  
[BugLab_Wrong_Literal]^return false;^178^^^^^175^181^return true;^[CLASS] Elements  [METHOD] hasText [RETURN_TYPE] boolean   [VARIABLES] List  contents  elements  Element  element  boolean  
[BugLab_Wrong_Literal]^return true;^180^^^^^175^181^return false;^[CLASS] Elements  [METHOD] hasText [RETURN_TYPE] boolean   [VARIABLES] List  contents  elements  Element  element  boolean  
[BugLab_Wrong_Operator]^if  ( sb.length (  )  == 0 ) sb.append ( "\n" ) ;^192^193^^^^189^197^if  ( sb.length (  )  != 0 ) sb.append ( "\n" ) ;^[CLASS] Elements  [METHOD] html [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  StringBuilder  sb  Element  element  
[BugLab_Wrong_Literal]^if  ( sb.length (  )  != 4 ) sb.append ( "\n" ) ;^192^193^^^^189^197^if  ( sb.length (  )  != 0 ) sb.append ( "\n" ) ;^[CLASS] Elements  [METHOD] html [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  StringBuilder  sb  Element  element  
[BugLab_Wrong_Operator]^if  ( sb.length (  )  == 0 ) sb.append ( "\n" ) ;^208^209^^^^205^213^if  ( sb.length (  )  != 0 ) sb.append ( "\n" ) ;^[CLASS] Elements  [METHOD] outerHtml [RETURN_TYPE] String   [VARIABLES] List  contents  elements  boolean  StringBuilder  sb  Element  element  
[BugLab_Argument_Swapping]^if  ( index.size (  )  > contents ) return new Elements ( get ( index )  ) ;^288^289^^^^287^292^if  ( contents.size (  )  > index ) return new Elements ( get ( index )  ) ;^[CLASS] Elements  [METHOD] eq [RETURN_TYPE] Elements   int index [VARIABLES] List  contents  elements  int  index  boolean  
[BugLab_Wrong_Operator]^if  ( contents.size (  )  >= index ) return new Elements ( get ( index )  ) ;^288^289^^^^287^292^if  ( contents.size (  )  > index ) return new Elements ( get ( index )  ) ;^[CLASS] Elements  [METHOD] eq [RETURN_TYPE] Elements   int index [VARIABLES] List  contents  elements  int  index  boolean  
[BugLab_Wrong_Literal]^return !contents.isEmpty (  )  ? contents.get ( -1 )  : null;^310^^^^^309^311^return !contents.isEmpty (  )  ? contents.get ( 0 )  : null;^[CLASS] Elements  [METHOD] first [RETURN_TYPE] Element   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Operator]^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )   >>  1 )  : null;^318^^^^^317^319^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )  - 1 )  : null;^[CLASS] Elements  [METHOD] last [RETURN_TYPE] Element   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Literal]^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )   )  : null;^318^^^^^317^319^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )  - 1 )  : null;^[CLASS] Elements  [METHOD] last [RETURN_TYPE] Element   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Operator]^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )   <<  1 )  : null;^318^^^^^317^319^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )  - 1 )  : null;^[CLASS] Elements  [METHOD] last [RETURN_TYPE] Element   [VARIABLES] List  contents  elements  boolean  
[BugLab_Wrong_Literal]^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )  -  )  : null;^318^^^^^317^319^return !contents.isEmpty (  )  ? contents.get ( contents.size (  )  - 1 )  : null;^[CLASS] Elements  [METHOD] last [RETURN_TYPE] Element   [VARIABLES] List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean contains ( Object o )  {return this.contains ( o ) ;}^326^^^^^321^331^public boolean contains ( Object o )  {return contents.contains ( o ) ;}^[CLASS] Elements  [METHOD] contains [RETURN_TYPE] boolean   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Argument_Swapping]^public boolean contains ( Object contents )  {return o.contains ( o ) ;}^326^^^^^321^331^public boolean contains ( Object o )  {return contents.contains ( o ) ;}^[CLASS] Elements  [METHOD] contains [RETURN_TYPE] boolean   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Variable_Misuse]^public Iterator<Element> iterator (  )  {return null.iterator (  ) ;}^328^^^^^323^333^public Iterator<Element> iterator (  )  {return contents.iterator (  ) ;}^[CLASS] Elements  [METHOD] iterator [RETURN_TYPE] Iterator   [VARIABLES] List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public <T> T[] toArray ( T[] contents )  {return a.toArray ( a ) ;}^332^^^^^327^337^public <T> T[] toArray ( T[] a )  {return contents.toArray ( a ) ;}^[CLASS] Elements  [METHOD] toArray [RETURN_TYPE] <T>   T[] a [VARIABLES] List  contents  elements  T[]  a  boolean  
[BugLab_Argument_Swapping]^public boolean add ( Element contents )  {return element.add ( element ) ;}^334^^^^^329^339^public boolean add ( Element element )  {return contents.add ( element ) ;}^[CLASS] Elements  [METHOD] add [RETURN_TYPE] boolean   Element element [VARIABLES] List  contents  elements  Element  element  boolean  
[BugLab_Argument_Swapping]^public boolean remove ( Object contents )  {return o.remove ( o ) ;}^336^^^^^331^341^public boolean remove ( Object o )  {return contents.remove ( o ) ;}^[CLASS] Elements  [METHOD] remove [RETURN_TYPE] boolean   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Variable_Misuse]^public boolean containsAll ( Collection<?> c )  {return null.containsAll ( c ) ;}^338^^^^^333^343^public boolean containsAll ( Collection<?> c )  {return contents.containsAll ( c ) ;}^[CLASS] Elements  [METHOD] containsAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean contentsontainsAll ( Collection<?> c )  {return c.containsAll ( c ) ;}^338^^^^^333^343^public boolean containsAll ( Collection<?> c )  {return contents.containsAll ( c ) ;}^[CLASS] Elements  [METHOD] containsAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean addAll ( Collection<? extends Element> 1 )  {return contents.addAll ( c ) ;}^340^^^^^335^345^public boolean addAll ( Collection<? extends Element> c )  {return contents.addAll ( c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean addAll ( Collection<? extends Element> contents )  {return c.addAll ( c ) ;}^340^^^^^335^345^public boolean addAll ( Collection<? extends Element> c )  {return contents.addAll ( c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean addAll ( Collection<? extends Element> c )  {return 4.addAll ( c ) ;}^340^^^^^335^345^public boolean addAll ( Collection<? extends Element> c )  {return contents.addAll ( c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean addAll ( int contents, Collection<? extends Element> c )  {return index.addAll ( index, c ) ;}^342^^^^^337^347^public boolean addAll ( int index, Collection<? extends Element> c )  {return contents.addAll ( index, c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   int index Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  int  index  
[BugLab_Argument_Swapping]^public boolean addAll ( int c, Collection<? extends Element> index )  {return contents.addAll ( index, c ) ;}^342^^^^^337^347^public boolean addAll ( int index, Collection<? extends Element> c )  {return contents.addAll ( index, c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   int index Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  int  index  
[BugLab_Argument_Swapping]^public boolean addAll ( int index, Collection<? extends Element> contents )  {return c.addAll ( index, c ) ;}^342^^^^^337^347^public boolean addAll ( int index, Collection<? extends Element> c )  {return contents.addAll ( index, c ) ;}^[CLASS] Elements  [METHOD] addAll [RETURN_TYPE] boolean   int index Element> c [VARIABLES] Collection  c  List  contents  elements  boolean  int  index  
[BugLab_Variable_Misuse]^public boolean removeAll ( Collection<?> c )  {return 0.removeAll ( c ) ;}^344^^^^^339^349^public boolean removeAll ( Collection<?> c )  {return contents.removeAll ( c ) ;}^[CLASS] Elements  [METHOD] removeAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean removeAll ( Collection<?> contents )  {return c.removeAll ( c ) ;}^344^^^^^339^349^public boolean removeAll ( Collection<?> c )  {return contents.removeAll ( c ) ;}^[CLASS] Elements  [METHOD] removeAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean removeAll ( Collection<?> 3 )  {return contents.removeAll ( c ) ;}^344^^^^^339^349^public boolean removeAll ( Collection<?> c )  {return contents.removeAll ( c ) ;}^[CLASS] Elements  [METHOD] removeAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean removeAll ( Collection<?> c )  {return 1.removeAll ( c ) ;}^344^^^^^339^349^public boolean removeAll ( Collection<?> c )  {return contents.removeAll ( c ) ;}^[CLASS] Elements  [METHOD] removeAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Variable_Misuse]^public boolean retainAll ( Collection<?> c )  {return null.retainAll ( c ) ;}^346^^^^^341^351^public boolean retainAll ( Collection<?> c )  {return contents.retainAll ( c ) ;}^[CLASS] Elements  [METHOD] retainAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean retainAll ( Collection<?> contents )  {return c.retainAll ( c ) ;}^346^^^^^341^351^public boolean retainAll ( Collection<?> c )  {return contents.retainAll ( c ) ;}^[CLASS] Elements  [METHOD] retainAll [RETURN_TYPE] boolean   Collection<?> c [VARIABLES] Collection  c  List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public boolean equals ( Object contents )  {return o.equals ( o ) ;}^350^^^^^345^355^public boolean equals ( Object o )  {return contents.equals ( o ) ;}^[CLASS] Elements  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Argument_Swapping]^public Element get ( int contents )  {return index.get ( index ) ;}^354^^^^^349^359^public Element get ( int index )  {return contents.get ( index ) ;}^[CLASS] Elements  [METHOD] get [RETURN_TYPE] Element   int index [VARIABLES] List  contents  elements  int  index  boolean  
[BugLab_Argument_Swapping]^public Element set ( int contents, Element element )  {return index.set ( index, element ) ;}^356^^^^^351^361^public Element set ( int index, Element element )  {return contents.set ( index, element ) ;}^[CLASS] Elements  [METHOD] set [RETURN_TYPE] Element   int index Element element [VARIABLES] List  contents  elements  boolean  int  index  Element  element  
[BugLab_Argument_Swapping]^public Element set ( int index, Element contents )  {return element.set ( index, element ) ;}^356^^^^^351^361^public Element set ( int index, Element element )  {return contents.set ( index, element ) ;}^[CLASS] Elements  [METHOD] set [RETURN_TYPE] Element   int index Element element [VARIABLES] List  contents  elements  boolean  int  index  Element  element  
[BugLab_Argument_Swapping]^public Element set ( int element, Element index )  {return contents.set ( index, element ) ;}^356^^^^^351^361^public Element set ( int index, Element element )  {return contents.set ( index, element ) ;}^[CLASS] Elements  [METHOD] set [RETURN_TYPE] Element   int index Element element [VARIABLES] List  contents  elements  boolean  int  index  Element  element  
[BugLab_Argument_Swapping]^public void add ( int element, Element index )  {contents.add ( index, element ) ;}^358^^^^^353^363^public void add ( int index, Element element )  {contents.add ( index, element ) ;}^[CLASS] Elements  [METHOD] add [RETURN_TYPE] void   int index Element element [VARIABLES] List  contents  elements  boolean  int  index  Element  element  
[BugLab_Argument_Swapping]^public Element remove ( int contents )  {return index.remove ( index ) ;}^360^^^^^355^365^public Element remove ( int index )  {return contents.remove ( index ) ;}^[CLASS] Elements  [METHOD] remove [RETURN_TYPE] Element   int index [VARIABLES] List  contents  elements  int  index  boolean  
[BugLab_Argument_Swapping]^public int indexOf ( Object contents )  {return o.indexOf ( o ) ;}^362^^^^^357^367^public int indexOf ( Object o )  {return contents.indexOf ( o ) ;}^[CLASS] Elements  [METHOD] indexOf [RETURN_TYPE] int   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Argument_Swapping]^public int lastIndexOf ( Object contents )  {return o.lastIndexOf ( o ) ;}^364^^^^^359^369^public int lastIndexOf ( Object o )  {return contents.lastIndexOf ( o ) ;}^[CLASS] Elements  [METHOD] lastIndexOf [RETURN_TYPE] int   Object o [VARIABLES] List  contents  elements  Object  o  boolean  
[BugLab_Variable_Misuse]^public ListIterator<Element> listIterator (  )  {return this.listIterator (  ) ;}^366^^^^^361^371^public ListIterator<Element> listIterator (  )  {return contents.listIterator (  ) ;}^[CLASS] Elements  [METHOD] listIterator [RETURN_TYPE] ListIterator   [VARIABLES] List  contents  elements  boolean  
[BugLab_Argument_Swapping]^public ListIterator<Element> listIterator ( int contents )  {return index.listIterator ( index ) ;}^368^^^^^363^373^public ListIterator<Element> listIterator ( int index )  {return contents.listIterator ( index ) ;}^[CLASS] Elements  [METHOD] listIterator [RETURN_TYPE] ListIterator   int index [VARIABLES] List  contents  elements  int  index  boolean  
[BugLab_Variable_Misuse]^public List<Element> subList ( int toIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^370^^^^^365^375^public List<Element> subList ( int fromIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^[CLASS] Elements  [METHOD] subList [RETURN_TYPE] List   int fromIndex int toIndex [VARIABLES] List  contents  elements  int  fromIndex  toIndex  boolean  
[BugLab_Argument_Swapping]^public List<Element> subList ( int contents, int toIndex )  {return fromIndex.subList ( fromIndex, toIndex ) ;}^370^^^^^365^375^public List<Element> subList ( int fromIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^[CLASS] Elements  [METHOD] subList [RETURN_TYPE] List   int fromIndex int toIndex [VARIABLES] List  contents  elements  int  fromIndex  toIndex  boolean  
[BugLab_Argument_Swapping]^public List<Element> subList ( int fromIndex, int contents )  {return toIndex.subList ( fromIndex, toIndex ) ;}^370^^^^^365^375^public List<Element> subList ( int fromIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^[CLASS] Elements  [METHOD] subList [RETURN_TYPE] List   int fromIndex int toIndex [VARIABLES] List  contents  elements  int  fromIndex  toIndex  boolean  
[BugLab_Variable_Misuse]^public List<Element> subList ( int fromIndex, int fromIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^370^^^^^365^375^public List<Element> subList ( int fromIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^[CLASS] Elements  [METHOD] subList [RETURN_TYPE] List   int fromIndex int toIndex [VARIABLES] List  contents  elements  int  fromIndex  toIndex  boolean  
[BugLab_Argument_Swapping]^public List<Element> subList ( int toIndex, int fromIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^370^^^^^365^375^public List<Element> subList ( int fromIndex, int toIndex )  {return contents.subList ( fromIndex, toIndex ) ;}^[CLASS] Elements  [METHOD] subList [RETURN_TYPE] List   int fromIndex int toIndex [VARIABLES] List  contents  elements  int  fromIndex  toIndex  boolean  
