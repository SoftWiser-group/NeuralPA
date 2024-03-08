[BugLab_Argument_Swapping]^this ( new XmlFactory ( outF, inputF )  ) ;^56^^^^^55^57^this ( new XmlFactory ( inputF, outF )  ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] XMLOutputFactory)   XMLInputFactory inputF XMLOutputFactory outF [VARIABLES] XMLOutputFactory  outF  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XMLInputFactory  inputF  
[BugLab_Variable_Misuse]^this ( xmlFactory, _xmlModule ) ;^67^^^^^66^68^this ( xmlFactory, DEFAULT_XML_MODULE ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] XmlFactory)   XmlFactory xmlFactory [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Argument_Swapping]^this ( DEFAULT_XML_MODULE, xmlFactory ) ;^67^^^^^66^68^this ( xmlFactory, DEFAULT_XML_MODULE ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] XmlFactory)   XmlFactory xmlFactory [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Variable_Misuse]^this ( new XmlFactory (  ) , _xmlModule ) ;^71^^^^^70^72^this ( new XmlFactory (  ) , module ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^_xmlModule = _xmlModule;^80^^^^^74^87^_xmlModule = module;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   XmlFactory xmlFactory JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Variable_Misuse]^if  ( _xmlModule != null )  {^82^^^^^74^87^if  ( module != null )  {^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   XmlFactory xmlFactory JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Wrong_Operator]^if  ( module == null )  {^82^^^^^74^87^if  ( module != null )  {^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   XmlFactory xmlFactory JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Variable_Misuse]^registerModule ( _xmlModule ) ;^83^^^^^74^87^registerModule ( module ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   XmlFactory xmlFactory JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Argument_Swapping]^_serializationConfig = DEFAULT_XML_PRETTY_PRINTER.withDefaultPrettyPrinter ( _serializationConfig ) ;^86^^^^^74^87^_serializationConfig = _serializationConfig.withDefaultPrettyPrinter ( DEFAULT_XML_PRETTY_PRINTER ) ;^[CLASS] XmlMapper  [METHOD] <init> [RETURN_TYPE] JacksonXmlModule)   XmlFactory xmlFactory JacksonXmlModule module [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  XmlFactory  xmlFactory  
[BugLab_Variable_Misuse]^return new XmlMapper (  ( XmlFactory )  _jsonFactory.copy (  ) , module ) ;^93^^^^^90^94^return new XmlMapper (  ( XmlFactory )  _jsonFactory.copy (  ) , _xmlModule ) ;^[CLASS] XmlMapper  [METHOD] copy [RETURN_TYPE] XmlMapper   [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new XmlMapper (  ( XmlFactory )  _xmlModule.copy (  ) , _jsonFactory ) ;^93^^^^^90^94^return new XmlMapper (  ( XmlFactory )  _jsonFactory.copy (  ) , _xmlModule ) ;^[CLASS] XmlMapper  [METHOD] copy [RETURN_TYPE] XmlMapper   [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( ai0 instanceof XmlAnnotationIntrospector )  {^126^^^^^122^131^if  ( ai instanceof XmlAnnotationIntrospector )  {^[CLASS] XmlMapper  [METHOD] setDefaultUseWrapper [RETURN_TYPE] XmlMapper   boolean state [VARIABLES] AnnotationIntrospector  ai  ai0  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  state  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ai  <=  XmlAnnotationIntrospector )  {^126^^^^^122^131^if  ( ai instanceof XmlAnnotationIntrospector )  {^[CLASS] XmlMapper  [METHOD] setDefaultUseWrapper [RETURN_TYPE] XmlMapper   boolean state [VARIABLES] AnnotationIntrospector  ai  ai0  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  state  long  serialVersionUID  
[BugLab_Variable_Misuse]^for  ( AnnotationIntrospector ai : ai.allIntrospectors (  )  )  {^125^^^^^122^131^for  ( AnnotationIntrospector ai : ai0.allIntrospectors (  )  )  {^[CLASS] XmlMapper  [METHOD] setDefaultUseWrapper [RETURN_TYPE] XmlMapper   boolean state [VARIABLES] AnnotationIntrospector  ai  ai0  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  state  long  serialVersionUID  
[BugLab_Argument_Swapping]^(  ( XmlFactory ) _jsonFactory ) .configure ( state, f ) ;^145^^^^^144^147^(  ( XmlFactory ) _jsonFactory ) .configure ( f, state ) ;^[CLASS] XmlMapper  [METHOD] configure [RETURN_TYPE] ObjectMapper   Feature f boolean state [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  state  Feature  f  long  serialVersionUID  
[BugLab_Argument_Swapping]^(  ( XmlFactory ) _jsonFactory ) .configure ( state, f ) ;^150^^^^^149^152^(  ( XmlFactory ) _jsonFactory ) .configure ( f, state ) ;^[CLASS] XmlMapper  [METHOD] configure [RETURN_TYPE] ObjectMapper   Feature f boolean state [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  boolean  state  Feature  f  long  serialVersionUID  
[BugLab_Argument_Swapping]^return _typeFactoryeadValue ( r, r.constructType ( valueType )  ) ;^188^^^^^187^189^return readValue ( r, _typeFactory.constructType ( valueType )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r Class<T> valueType [VARIABLES] Class  valueType  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return readValue ( r, valueType.constructType ( _typeFactory )  ) ;^188^^^^^187^189^return readValue ( r, _typeFactory.constructType ( valueType )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r Class<T> valueType [VARIABLES] Class  valueType  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return valueTypeeadValue ( r, _typeFactory.constructType ( r )  ) ;^188^^^^^187^189^return readValue ( r, _typeFactory.constructType ( valueType )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r Class<T> valueType [VARIABLES] Class  valueType  JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return valueTypeRefeadValue ( r, _typeFactory.constructType ( r )  ) ;^199^^^^^198^200^return readValue ( r, _typeFactory.constructType ( valueTypeRef )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r TypeReference<T> valueTypeRef [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  TypeReference  valueTypeRef  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return readValue ( r, valueTypeRef.constructType ( _typeFactory )  ) ;^199^^^^^198^200^return readValue ( r, _typeFactory.constructType ( valueTypeRef )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r TypeReference<T> valueTypeRef [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  TypeReference  valueTypeRef  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return _typeFactoryeadValue ( r, r.constructType ( valueTypeRef )  ) ;^199^^^^^198^200^return readValue ( r, _typeFactory.constructType ( valueTypeRef )  ) ;^[CLASS] XmlMapper  [METHOD] readValue [RETURN_TYPE] <T>   XMLStreamReader r TypeReference<T> valueTypeRef [VARIABLES] JacksonXmlModule  DEFAULT_XML_MODULE  _xmlModule  module  DefaultXmlPrettyPrinter  DEFAULT_XML_PRETTY_PRINTER  XMLStreamReader  r  TypeReference  valueTypeRef  boolean  long  serialVersionUID  