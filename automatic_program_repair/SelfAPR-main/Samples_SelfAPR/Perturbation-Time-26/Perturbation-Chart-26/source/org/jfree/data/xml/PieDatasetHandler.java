[P8_Replace_Mix]^this.dataset = this;^63^^^^^62^64^this.dataset = null;^[CLASS] PieDatasetHandler  [METHOD] <init> [RETURN_TYPE] PieDatasetHandler()   [VARIABLES] DefaultPieDataset  dataset  boolean  
[P8_Replace_Mix]^return dataset;^72^^^^^71^73^return this.dataset;^[CLASS] PieDatasetHandler  [METHOD] getDataset [RETURN_TYPE] PieDataset   [VARIABLES] DefaultPieDataset  dataset  boolean  
[P5_Replace_Variable]^this.dataset.setValue (  value ) ;^82^^^^^81^83^this.dataset.setValue ( key, value ) ;^[CLASS] PieDatasetHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] DefaultPieDataset  dataset  Comparable  key  boolean  Number  value  
[P5_Replace_Variable]^this.dataset.setValue ( key ) ;^82^^^^^81^83^this.dataset.setValue ( key, value ) ;^[CLASS] PieDatasetHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] DefaultPieDataset  dataset  Comparable  key  boolean  Number  value  
[P5_Replace_Variable]^this.dataset.setValue ( value, key ) ;^82^^^^^81^83^this.dataset.setValue ( key, value ) ;^[CLASS] PieDatasetHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] DefaultPieDataset  dataset  Comparable  key  boolean  Number  value  
[P14_Delete_Statement]^^82^^^^^81^83^this.dataset.setValue ( key, value ) ;^[CLASS] PieDatasetHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] DefaultPieDataset  dataset  Comparable  key  boolean  Number  value  
[P7_Replace_Invocation]^DefaultHandler current = getSubHandlers (  ) ;^100^^^^^95^113^DefaultHandler current = getCurrentHandler (  ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P14_Delete_Statement]^^100^^^^^95^113^DefaultHandler current = getCurrentHandler (  ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P2_Replace_Operator]^if  ( current == this )  {^101^^^^^95^113^if  ( current != this )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P15_Unwrap_Block]^current.startElement(namespaceURI, localName, qName, atts);^101^102^103^^^95^113^if  ( current != this )  { current.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P16_Remove_Block]^^101^102^103^^^95^113^if  ( current != this )  { current.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P13_Insert_Block]^if  ( qName.equals ( PIEDATASET_TAG )  )  {     this.dataset = new DefaultPieDataset (  ) ; }else     if  ( qName.equals ( ITEM_TAG )  )  {         ItemHandler subhandler = new ItemHandler ( this, this ) ;         getSubHandlers (  ) .push ( subhandler ) ;         subhandler.startElement ( namespaceURI, localName, qName, atts ) ;     }^101^^^^^95^113^[Delete]^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^else if  ( namespaceURI.equals ( PIEDATASET_TAG )  )  {^104^^^^^95^113^else if  ( qName.equals ( PIEDATASET_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^if  ( qName.equals ( PIEDATASET_TAG )  )  {^104^^^^^95^113^else if  ( qName.equals ( PIEDATASET_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P9_Replace_Statement]^else if  ( qName.equals ( ITEM_TAG )  )  {^104^^^^^95^113^else if  ( qName.equals ( PIEDATASET_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P15_Unwrap_Block]^this.dataset = new org.jfree.data.general.DefaultPieDataset();^104^105^106^^^95^113^else if  ( qName.equals ( PIEDATASET_TAG )  )  { this.dataset = new DefaultPieDataset (  ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P16_Remove_Block]^^104^105^106^^^95^113^else if  ( qName.equals ( PIEDATASET_TAG )  )  { this.dataset = new DefaultPieDataset (  ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^else if  ( namespaceURI.equals ( ITEM_TAG )  )  {^107^^^^^95^113^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^else {^107^^^^^95^113^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P9_Replace_Statement]^else if  ( qName.equals ( PIEDATASET_TAG )  )  {^107^^^^^95^113^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P15_Unwrap_Block]^org.jfree.data.xml.ItemHandler subhandler = new org.jfree.data.xml.ItemHandler(this, this); getSubHandlers().push(subhandler); subhandler.startElement(namespaceURI, localName, qName, atts);^107^108^109^110^111^95^113^else if  ( qName.equals ( ITEM_TAG )  )  { ItemHandler subhandler = new ItemHandler ( this, this ) ; getSubHandlers (  ) .push ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P16_Remove_Block]^^107^108^109^110^111^95^113^else if  ( qName.equals ( ITEM_TAG )  )  { ItemHandler subhandler = new ItemHandler ( this, this ) ; getSubHandlers (  ) .push ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P7_Replace_Invocation]^getCurrentHandler (  ) .push ( subhandler ) ;^109^^^^^95^113^getSubHandlers (  ) .push ( subhandler ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P14_Delete_Statement]^^109^110^^^^95^113^getSubHandlers (  ) .push ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P14_Delete_Statement]^^109^^^^^95^113^getSubHandlers (  ) .push ( subhandler ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( qName, localName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, qName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, namespaceURI, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement (  localName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI,  qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName,  atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, qName ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( qName, localName, namespaceURI, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, qName, localName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, atts, qName, localName ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P14_Delete_Statement]^^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P11_Insert_Donor_Statement]^current.startElement ( namespaceURI, localName, qName, atts ) ;subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P11_Insert_Donor_Statement]^current.endElement ( namespaceURI, localName, qName ) ;subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, namespaceURI, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, atts, qName ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^subhandler.startElement ( localName, localName, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^this.dataset ;^105^^^^^95^113^this.dataset = new DefaultPieDataset (  ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P7_Replace_Invocation]^getSubHandlers (  )  .getSubHandlers (  )  ;^109^^^^^95^113^getSubHandlers (  ) .push ( subhandler ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, localName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( qName, localName, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, qName, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, localName, namespaceURI, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement (  localName, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI,  qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, localName,  atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, localName, qName ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( localName, namespaceURI, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, qName, localName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( atts, localName, qName, namespaceURI ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P7_Replace_Invocation]^current .endElement ( localName , namespaceURI , namespaceURI )  ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^current.startElement ( namespaceURI, namespaceURI, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P14_Delete_Statement]^^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P11_Insert_Donor_Statement]^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;current.startElement ( namespaceURI, localName, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P11_Insert_Donor_Statement]^current.endElement ( namespaceURI, localName, qName ) ;current.startElement ( namespaceURI, localName, qName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P13_Insert_Block]^if  ( current !=  ( this )  )  {     current.endElement ( namespaceURI, localName, qName ) ; }^102^^^^^95^113^[Delete]^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( localName, namespaceURI, qName, atts ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^subhandler.startElement ( atts, localName, qName, namespaceURI ) ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^this.dataset  =  this.dataset ;^105^^^^^95^113^this.dataset = new DefaultPieDataset (  ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^if  ( namespaceURI.equals ( ITEM_TAG )  )  {^107^^^^^95^113^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^subhandler .endElement ( qName , localName , localName )  ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P5_Replace_Variable]^current.startElement ( namespaceURI, localName, atts, qName ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P8_Replace_Mix]^current.startElement ( namespaceURI, localName, localName, atts ) ;^102^^^^^95^113^current.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P7_Replace_Invocation]^subhandler .endElement ( qName , namespaceURI , qName )  ;^110^^^^^95^113^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] PieDatasetHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Attributes  atts  DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  ItemHandler  subhandler  DefaultHandler  current  
[P7_Replace_Invocation]^DefaultHandler current = getSubHandlers (  ) ;^128^^^^^124^133^DefaultHandler current = getCurrentHandler (  ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P14_Delete_Statement]^^128^^^^^124^133^DefaultHandler current = getCurrentHandler (  ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P2_Replace_Operator]^if  ( current <= this )  {^129^^^^^124^133^if  ( current != this )  {^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P15_Unwrap_Block]^current.endElement(namespaceURI, localName, qName);^129^130^131^^^124^133^if  ( current != this )  { current.endElement ( namespaceURI, localName, qName ) ; }^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P16_Remove_Block]^^129^130^131^^^124^133^if  ( current != this )  { current.endElement ( namespaceURI, localName, qName ) ; }^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( qName, localName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI, namespaceURI, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI, localName, namespaceURI ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement (  localName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI,  qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI, localName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( localName, namespaceURI, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI, qName, localName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( qName, localName, namespaceURI ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P14_Delete_Statement]^^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P11_Insert_Donor_Statement]^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;current.endElement ( namespaceURI, localName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P11_Insert_Donor_Statement]^current.startElement ( namespaceURI, localName, qName, atts ) ;current.endElement ( namespaceURI, localName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P13_Insert_Block]^if  ( current !=  ( this )  )  {     current.endElement ( namespaceURI, localName, qName ) ; }^130^^^^^124^133^[Delete]^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( localName, localName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  
[P5_Replace_Variable]^current.endElement ( namespaceURI, qName, qName ) ;^130^^^^^124^133^current.endElement ( namespaceURI, localName, qName ) ;^[CLASS] PieDatasetHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] DefaultPieDataset  dataset  String  localName  namespaceURI  qName  boolean  DefaultHandler  current  